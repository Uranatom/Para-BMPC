"""Tree AL-iLQR and CVaR-SOCP solvers for the declared common-model adapters.

These are paper-derived adapters, not renamed shooting/LTV implementations.
The independent original-source numerical checks live in native_* modules.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse
import ecos


def project_capped_simplex(point,upper):
    point=np.asarray(point,float);upper=np.asarray(upper,float)
    if point.shape!=upper.shape or np.any(upper<0) or upper.sum()<1-1e-12:
        raise ValueError('Invalid probability envelope.')
    lo=float(np.min(point-upper))-1.;hi=float(np.max(point))+1.
    for _ in range(80):
        mid=.5*(lo+hi)
        if np.clip(point-mid,0,upper).sum()>1:lo=mid
        else:hi=mid
    return np.clip(point-.5*(lo+hi),0,upper)


def risk_vertices(p,tau):
    p=np.asarray(p,float)
    if p.shape!=(2,) or np.any(p<0) or abs(p.sum()-1)>1e-6 or not 0<tau<=1:
        raise ValueError('Two normalized probabilities and a positive tail mass are required.')
    p=p/p.sum()
    low=max(0.,1-p[1]/tau);high=min(1.,p[0]/tau)
    return np.array([[low,1-low],[high,1-high]])


@dataclass
class SolverSettings:
    method: str
    tail_mass: float=.1
    shared_prefix: int=1
    iterations: int=8
    outer_iterations: int=2
    regularization: float=1e-3
    rho: float=100.
    rho_scaling: float=10.
    probability_step: float=.008
    probability_regularization: float=500.
    trust_accel: float=.8
    trust_steer_rate: float=.08
    weight_scale: float=1.
    probability_model: str='shared_observation'
    probability_schedule: str='published_inverse_sqrt'
    lateral_weight: float=1.
    heading_weight: float=1.
    steering_weight: float=1.
    terminal_scale: float=1.
    road_buffer_m: float=0.
    constraint_penalty: float=500.
    sat_target_m: float=1.5

    def __post_init__(self):
        if self.method not in ('era','branch') or not 0<self.tail_mass<=1:
            raise ValueError('Invalid method or tail mass.')
        if not 1<=self.shared_prefix<10 or self.iterations<1 or self.outer_iterations<1:
            raise ValueError('Invalid shared prefix or iteration budget.')
        if self.probability_model not in ('shared_observation','native_safety'):
            raise ValueError('Invalid predictor variant.')
        if self.probability_schedule not in ('published_inverse_sqrt','preprint_inverse_linear'):
            raise ValueError('Unknown regularization schedule.')
        if min(self.lateral_weight,self.heading_weight,self.steering_weight,self.terminal_scale,self.constraint_penalty)<=0 or self.road_buffer_m<0 or self.sat_target_m<1.5:
            raise ValueError('Positive cost weights and nonnegative road buffer are required.')


class TreeILQR:
    def __init__(self,model,settings):
        self.model=model;self.settings=settings;self.N=model.config.horizon
        self.maps=[function.map(self.N,'serial') for function in model.derivatives]
        self.fj=model.fj.map(self.N,'serial')

    def rollout(self,initial,controls):
        states=np.zeros((2,self.N+1,self.model.nx));states[:,0]=initial
        for b in range(2):
            for k in range(self.N):states[b,k+1]=np.asarray(self.model.f(states[b,k],controls[b,k])).ravel()
        return states

    def evaluate(self,states,controls,start,mu,rho,q,derivatives=False,p=None):
        total=0.;costs=np.zeros(2);hs=[];terms=[]
        for b in range(2):
            r,J,h,H=[np.asarray(v) for v in self.maps[b](states[b,:-1].T,controls[b].T,
                np.arange(self.N)[None,:],np.full((1,self.N),start))]
            weights=np.full(self.N,q[b])
            if p is not None:weights[:self.settings.shared_prefix]=p[b]
            costs[b]=float(np.sum(r[:,self.settings.shared_prefix:]**2));hs.append(h.T)
            active=(h.T>0)|(mu[b]>0)
            total+=float(np.sum(weights[None,:]*r*r))+float(np.sum(np.where(active,mu[b]*h.T+.5*rho*h.T**2,0)))
            if derivatives:
                local=[]
                for k in range(self.N):
                    jr=J[:,k*20:(k+1)*20];jh=H[:,k*20:(k+1)*20]
                    gradient=2*weights[k]*jr.T@r[:,k]+jh.T@np.where(active[k],mu[b,k]+rho*h[:,k],0)
                    hessian=2*weights[k]*jr.T@jr+rho*jh.T@(active[k,:,None]*jh)
                    local.append((gradient,hessian))
                terms.append(local)
        return total,costs,np.asarray(hs),terms

    def backward(self,states,controls,terms,regularization):
        nx=self.model.nx;N=self.N;s=self.settings.shared_prefix
        gains=np.zeros((2,N,2,nx));directions=np.zeros((2,N,2))
        V=np.zeros((2,N+1,nx,nx));vg=np.zeros((2,N+1,nx))
        models=[]
        for b in range(2):
            A,B=[np.asarray(v) for v in self.fj(states[b,:-1].T,controls[b].T)]
            models.append([(A[:,k*nx:(k+1)*nx],B[:,k*2:(k+1)*2]) for k in range(N)])

        def step(b,k,gradient,hessian,nextV,nextg):
            A,B=models[b][k];F=np.column_stack((A,B))
            H=hessian+F.T@nextV@F;g=gradient+F.T@nextg
            Q=H[nx:,nx:]+regularization*np.eye(2)
            K=-np.linalg.solve(Q,H[nx:,:nx]);d=-np.linalg.solve(Q,g[nx:])
            value=H[:nx,:nx]+K.T@H[nx:,nx:]@K+K.T@H[nx:,:nx]+H[:nx,nx:]@K
            valueg=g[:nx]+K.T@H[nx:,nx:]@d+K.T@g[nx:]+H[:nx,nx:]@d
            return K,d,.5*(value+value.T),valueg

        for b in range(2):
            for k in range(N-1,s-1,-1):
                gains[b,k],directions[b,k],V[b,k],vg[b,k]=step(b,k,*terms[b][k],V[b,k+1],vg[b,k+1])
        for k in range(s-1,-1,-1):
            nextV=V[:,k+1].sum(0) if k==s-1 else V[0,k+1]
            nextg=vg[:,k+1].sum(0) if k==s-1 else vg[0,k+1]
            gradient=terms[0][k][0]+terms[1][k][0];hessian=terms[0][k][1]+terms[1][k][1]
            K,d,value,valueg=step(0,k,gradient,hessian,nextV,nextg)
            gains[:,k]=K;directions[:,k]=d;V[0,k]=value;vg[0,k]=valueg
        return gains,directions

    def solve(self,initial,controls,p):
        settings=self.settings;start=self.model.start_margin(initial)
        controls=controls.copy();controls[:,:settings.shared_prefix]=controls[0,:settings.shared_prefix]
        states=self.rollout(initial,controls);mu=np.zeros((2,self.N,self.model.nh));rho=settings.rho
        q=np.asarray(p,float).copy();history=[];converged=False;iteration=0
        reg=settings.regularization
        for outer in range(settings.outer_iterations):
            for inner in range(settings.iterations):
                value,costs,h,terms=self.evaluate(states,controls,start,mu,rho,q,True,p)
                try:K,d=self.backward(states,controls,terms,reg)
                except np.linalg.LinAlgError:
                    reg*=10
                    history.append({'iteration':iteration,'factorization_failed':True});iteration+=1
                    continue
                accepted=False
                for alpha in (1.,.5,.25,.125,.0625):
                    newx=np.zeros_like(states);newx[:,0]=initial;newu=np.zeros_like(controls)
                    for b in range(2):
                        for k in range(self.N):
                            if b==1 and k<settings.shared_prefix:newu[b,k]=newu[0,k]
                            else:
                                command=controls[b,k]+alpha*d[b,k]+K[b,k]@(newx[b,k]-states[b,k])
                                v=self.model.vehicle
                                newu[b,k]=np.clip(command,[-v.max_brake_mps2,-v.max_steer_rate_rad_s],[v.max_accel_mps2,v.max_steer_rate_rad_s])
                            newx[b,k+1]=np.asarray(self.model.f(newx[b,k],newu[b,k])).ravel()
                    newvalue,newcost,newh,_=self.evaluate(newx,newu,start,mu,rho,q,p=p)
                    if np.isfinite(newvalue) and newvalue<=value+1e-9:
                        accepted=True;break
                if accepted:
                    improvement=value-newvalue;states,controls,costs,h=newx,newu,newcost,newh
                    reg=max(settings.regularization,reg*.5)
                else:improvement=0.;reg=min(1e6,reg*10)
                denominator=np.sqrt(iteration+1.) if settings.probability_schedule=='published_inverse_sqrt' else iteration+1.
                regularization=settings.probability_regularization/denominator
                # Algorithm 2 updates adverse weights only after an accepted
                # forward pass; a failed line search only raises regularization.
                nextq=(project_capped_simplex((1-settings.probability_step*regularization)*q+
                    settings.probability_step*costs,np.asarray(p)/settings.tail_mass)
                    if accepted else q.copy())
                qchange=float(np.max(np.abs(nextq-q)));q=nextq
                history.append({'iteration':iteration,'cost':float(np.dot(q,costs)),
                    'q':q.tolist(),'q_change':qchange,'max_constraint':float(np.maximum(h,0).max()),
                    'accepted':accepted,'probability_updated':accepted,'regularization':regularization})
                iteration+=1
                converged=accepted and improvement<=1e-4*(1+abs(value)) and qchange<=1e-3
                if converged:break
            mu=np.maximum(0.,mu+rho*h)
            if converged and np.maximum(h,0).max()<=1e-4:break
            rho*=settings.rho_scaling
        return controls,{'solver':'tree_al_ilqr','finite':bool(np.isfinite(controls).all()),
            'converged':bool(converged),'iterations':iteration,'q':q.tolist(),'history':history,
            'stationarity_claim':False}


class BranchSOCP:
    def __init__(self,model,settings):
        self.model=model;self.settings=settings;self.N=model.config.horizon
        self.function=model.condensed(settings.shared_prefix)

    def pack(self,controls):
        s=self.settings.shared_prefix
        return np.concatenate((controls[0,:s].ravel(),controls[0,s:].ravel(),controls[1,s:].ravel()))

    def unpack(self,point):
        s=self.settings.shared_prefix;N=self.N
        shared=point[:2*s].reshape(s,2);offset=2*s
        return np.array([np.vstack((shared,point[offset+b*2*(N-s):offset+(b+1)*2*(N-s)].reshape(N-s,2))) for b in range(2)])

    def solve(self,initial,controls,p):
        point=self.pack(controls);start=self.model.start_margin(initial);vertices=risk_vertices(p,self.settings.tail_mass)
        n=len(point);history=[];all_statuses=[];converged=False;penalty=self.settings.constraint_penalty;numerically_valid=True
        for iteration in range(self.settings.iterations):
            r0,r1,h0,h1,J0,J1,H0,H1=[np.asarray(a) for a in self.function(point,initial,start)]
            if not all(np.isfinite(a).all() for a in (r0,r1,h0,h1,J0,J1,H0,H1)):
                numerically_valid=False
                history.append({'iteration':iteration,'accepted':False,'failure':'nonfinite_linearization'})
                break
            rs=[r0.ravel(),r1.ravel()];hs=[h0.ravel(),h1.ravel()];Js=[J0,J1];Hs=[H0,H1]
            nh=len(h0);ns=2*nh;dimension=n+1+ns;t=n;slack=n+1
            # Hard linear inequalities, then risk epigraph cones.
            H=np.vstack(Hs);h=np.concatenate(hs)
            linear=np.zeros((2*ns+4*n,dimension));rhs=np.zeros(len(linear))
            linear[:ns,:n]=H;linear[:ns,slack:]=-np.eye(ns);rhs[:ns]=-h
            linear[ns:2*ns,slack:]=-np.eye(ns)
            trust=np.tile([self.settings.trust_accel,self.settings.trust_steer_rate],n//2)
            v=self.model.vehicle;low=np.tile([-v.max_brake_mps2,-v.max_steer_rate_rad_s],n//2);high=np.tile([v.max_accel_mps2,v.max_steer_rate_rad_s],n//2)
            for j,(sign,bound) in enumerate(((1,trust),(-1,trust),(1,high-point),(-1,point-low))):
                offset=2*ns+j*n;linear[offset:offset+n,:n]=sign*np.eye(n);rhs[offset:offset+n]=bound
            matrices=[linear];bounds=[rhs];cones=[]
            for q in vertices:
                A=np.zeros((len(rs[0])+len(rs[1])+ns+n,dimension));r=np.zeros(len(A));row=0
                for b in range(2):
                    length=len(rs[b]);stageweights=np.full(self.N,q[b]);stageweights[:self.settings.shared_prefix]=p[b]
                    scales=np.repeat(np.sqrt(stageweights),length//self.N)
                    A[row:row+length,:n]=scales[:,None]*Js[b];r[row:row+length]=scales*rs[b];row+=length
                slack_scales=[]
                for b in range(2):
                    stageweights=np.full(self.N,q[b]);stageweights[:self.settings.shared_prefix]=p[b]
                    slack_scales.extend(np.repeat(np.sqrt(penalty*stageweights),nh//self.N))
                A[row:row+ns,slack:]=np.diag(slack_scales);row+=ns
                A[row:,:n]=np.sqrt(self.settings.regularization)*np.eye(n)
                G=np.zeros((len(A)+2,dimension));g=np.zeros(len(G));G[0,t]=-1;g[0]=1
                G[1:-1]=-2*A;g[1:-1]=2*r;G[-1,t]=-1;g[-1]=-1
                matrices.append(G);bounds.append(g);cones.append(len(G))
            objective=np.zeros(dimension);objective[t]=1.
            solution=ecos.solve(objective,sparse.csc_matrix(np.vstack(matrices)),np.concatenate(bounds),
                {'l':len(linear),'q':cones,'e':0},verbose=False,max_iters=100,
                feastol=1e-7,abstol=1e-7,reltol=1e-7)
            info=solution['info'];all_statuses.append(int(info['exitFlag']))
            record={'iteration':iteration,'info':{k:float(info[k]) for k in ('exitFlag','pres','dres','gap','iter')}}
            solution_finite=solution['x'] is not None and bool(np.isfinite(solution['x']).all())
            record['solution_finite']=solution_finite
            if not solution_finite:numerically_valid=False
            if info['exitFlag'] not in (0,10) or not solution_finite:
                record['accepted']=False;history.append(record);break
            delta=solution['x'][:n]

            def score(x):
                values=[np.asarray(a).ravel() for a in self.function(x,initial,start)[:4]]
                prefix=0.;costs=[]
                for b in range(2):
                    r=values[b].reshape(self.N,-1);h=np.maximum(values[b+2],0).reshape(self.N,-1)
                    stages=np.sum(r*r,axis=1)+penalty*np.sum(h*h,axis=1)
                    prefix+=p[b]*sum(stages[:self.settings.shared_prefix]);costs.append(sum(stages[self.settings.shared_prefix:]))
                return float(prefix+np.max(vertices@np.asarray(costs)))

            current=score(point);accepted=False
            for alpha in (1.,.5,.25,.125,.0625):
                candidate=point+alpha*delta;value=score(candidate)
                if np.isfinite(value) and value<=current+1e-9:accepted=True;break
            record.update(accepted=accepted,cost=value,step_norm=float(np.max(np.abs(delta))))
            history.append(record)
            if accepted:point=candidate
            converged=accepted and (current-value<=1e-4*(1+abs(current)) or np.max(np.abs(delta))<1e-5)
            if converged:break
            if not accepted:break
        return self.unpack(point),{'solver':'sequential_cvar_socp','finite':bool(np.isfinite(point).all() and numerically_valid),
            'converged':bool(converged),'iterations':len(history),'exit_flags':all_statuses,
            'subproblem_accepted':bool(numerically_valid and all_statuses and all(s in (0,10) for s in all_statuses)),
            'history':history,'stationarity_claim':False}
