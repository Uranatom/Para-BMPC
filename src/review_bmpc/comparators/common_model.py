"""CasADi form of the frozen R7 plant, interaction and predictive constraints.

The augmented state contains ego, both hypothetical SV states, the previous
command and observed acceleration. Thus a shared control prefix propagates
one common information state without consulting the realized SV mode.
"""
from __future__ import annotations
import math
import casadi as ca
import numpy as np


def clip(x, low, high):
    return ca.fmin(ca.fmax(x, low), high)


def nonnegative_sqrt(x):
    """Preserve sqrt(max(x,0)); choose a finite zero derivative at the kink.

    The inner inactive argument stays positive so symbolic AD never forms
    0/sqrt(0), including inside an inactive reference-speed branch.
    """
    positive=x>0
    return ca.if_else(positive,ca.sqrt(ca.if_else(positive,x,1.)),0.)


def step_vehicle(x, u, v):
    dt=v.dt_s
    a=clip(u[0],-v.max_brake_mps2,v.max_accel_mps2)
    a=ca.fmax(a,-ca.fmax(x[3],0)/(2*dt*v.longitudinal_gain))
    a=ca.fmin(a,ca.fmax(v.max_speed_mps-x[3],0)/(2*dt*v.longitudinal_gain))
    rate=clip(u[1],-v.max_steer_rate_rad_s,v.max_steer_rate_rad_s)
    target=clip(x[4]+dt*rate,-v.max_steer_rad,v.max_steer_rad)
    applied=(target-x[4])/dt
    drag=v.rolling_decel_mps2+v.speed_drag_gain*x[3]**2+v.turning_drag_gain*x[3]**2*x[4]**2
    speed=clip(x[3]+dt*(v.longitudinal_gain*a-drag*ca.if_else(x[3]>0,1,0)),v.min_speed_mps,v.max_speed_mps)
    yaw_rate=x[3]/v.wheelbase_m*((v.steering_gain+v.yaw_speed_gain*x[3]**2)*ca.tan(x[4])+v.yaw_rate_response*dt*applied)
    steering_rate=v.steering_rate_gain*applied-(v.steering_decay_s+v.steering_speed_decay*x[3]**2)*x[4]
    return ca.vertcat(x[0]+dt*.5*(x[3]+speed)*ca.cos(x[2]),
        x[1]+dt*.5*(x[3]+speed)*ca.sin(x[2]),x[2]+dt*yaw_rate,speed,
        clip(x[4]+dt*steering_rate,-v.max_steer_rad,v.max_steer_rad))


def stop_distance(speed,v):
    return 1.5*speed*v.dt_s+speed**2/(2*v.max_brake_mps2*v.longitudinal_gain)


def other_action(name,z,e,branch,v):
    speed=z[3]; desired=9.5 if branch==0 else 7.
    acceleration=2*(1-(speed/desired)**4)
    if name=='merge':
        ec=e[0]+v.lr_m*ca.cos(e[2]);zc=z[0]+v.lr_m*ca.cos(z[2])
        phase=clip((ca.fabs(e[1]-z[1])-.75*v.body_width_m)/v.body_width_m,0,1)
        overlap=1-phase**2*(3-2*phase)
        net=ca.fmax(ec-zc-v.body_length_m,.1)
        desired_gap=(1. if branch==0 else 2.5)+ca.fmax(speed*(.7 if branch==0 else 1.6)+speed*(speed-e[3])/(2*math.sqrt(6)),0)
        acceleration-=ca.if_else(ec>zc,2*overlap*(desired_gap/net)**2,0)
        if branch==1:
            target=ca.if_else(z[0]<28,4.5,desired)
            acceleration=ca.fmin(acceleration,1.5*(target-speed))
    else:
        front=v.lr_m+.5*v.body_length_m;rear=v.lr_m-.5*v.body_length_m
        clear=e[0]+rear>2.55
        occupies=ca.logic_and(e[0]+front>-2.55,e[0]+rear<2.55)
        distance=ca.fmax(-2.55-front-3.5-z[1],0)
        decel=v.max_brake_mps2*v.longitudinal_gain
        target=ca.fmin(ca.sqrt((decel*v.dt_s)**2+2*decel*distance)-decel*v.dt_s,7.)
        can_stop=ca.fmax(-2.55-front-z[1],0)>=stop_distance(speed,v)
        must_yield=ca.logic_not(clear) if branch==1 else ca.logic_and(occupies,can_stop)
        acceleration=ca.if_else(must_yield,ca.fmin(acceleration,(target-speed)/v.dt_s),acceleration)
    return ca.vertcat(clip(acceleration,-v.max_brake_mps2,v.max_accel_mps2),0)


def axes(x):
    forward=ca.vertcat(ca.cos(x[2]),ca.sin(x[2]))
    return forward,ca.vertcat(-forward[1],forward[0])


def smooth_clearance(e,z,v):
    ef,el=axes(e);zf,zl=axes(z)
    displacement=e[:2]+v.lr_m*ef-z[:2]-v.lr_m*zf
    absolute=lambda value:ca.sqrt(value**2+.01**2)
    gaps=[]
    for axis in (ef,el,zf,zl):
        center=absolute(ca.dot(axis,displacement))-.01
        radius=.5*v.body_length_m*(absolute(ca.dot(axis,ef))+absolute(ca.dot(axis,zf)))
        radius+=.5*v.body_width_m*(absolute(ca.dot(axis,el))+absolute(ca.dot(axis,zl)))
        gaps.append(center-radius)
    return .05*ca.logsumexp(ca.vertcat(*gaps)/.05)-.05*math.log(4.)


def road_gaps(e,name,v):
    center=e[1]+v.lr_m*ca.sin(e[2])
    half=.5*v.body_length_m*ca.sqrt(ca.sin(e[2])**2+.01**2)+.5*v.body_width_m*ca.sqrt(ca.cos(e[2])**2+.01**2)
    low,high=(-1.75,5.25) if name=='merge' else (-1.75,1.75)
    return ca.vertcat(center-half-low,high-center-half)


def release_allowed(e,z,v):
    half=.5*v.body_length_m*ca.fabs(ca.sin(z[2]))+.5*v.body_width_m*ca.fabs(ca.cos(z[2]))
    center=z[1]+v.lr_m*ca.sin(z[2]);front=center+half;rear=center-half
    ef=e[0]+v.lr_m*ca.cos(e[2])+.5*v.body_length_m*ca.fabs(ca.cos(e[2]))+.5*v.body_width_m*ca.fabs(ca.sin(e[2]))
    ea=max(.1,.5*v.max_accel_mps2*v.longitudinal_gain);za=max(.1,v.max_accel_mps2*v.longitudinal_gain)
    entry=ca.fmax(-2.55-ef,0)
    entry_time=(nonnegative_sqrt(e[3]**2+2*ea*entry)-e[3])/ea
    zdistance=ca.fmax(-2.55-front,0)
    reach=(nonnegative_sqrt(z[3]**2+2*za*zdistance)-z[3])/za
    can_stop=ca.fmax(zdistance-2.,0)+1e-5>=stop_distance(z[3],v)
    precedence=ca.logic_or(ef>-2.55,ca.logic_and(z[3]<.3,entry_time+v.dt_s<reach))
    return ca.logic_or(rear>2.55,ca.logic_and(front<-2.55,ca.logic_and(can_stop,precedence)))


def reference(name,e,z,branch,v):
    s=clip((e[0]-8)/40,0,1)
    y=3.5*(10*s**3-15*s**4+6*s**5) if name=='merge' else 0.
    heading=ca.atan(3.5/40*30*s**2*(1-s)**2) if name=='merge' else 0.
    speed=8.
    if branch==0:
        if name=='merge':
            gap=z[0]-e[0]
            phase=ca.fmax(ca.fmax(clip((gap-16)/6,0,1),clip((-gap-7)/6,0,1)),clip((e[1]-.5)/.7,0,1))
            gate=phase**2*(3-2*phase)
            y=gate*y;heading=ca.atan(gate*ca.tan(heading))
            speed=ca.if_else(z[3]<=3.5,8.,3+5*gate)
        else:
            half=.5*v.body_length_m*ca.fabs(ca.sin(z[2]))+.5*v.body_width_m*ca.fabs(ca.cos(z[2]))
            clear=z[1]+v.lr_m*ca.sin(z[2])-half>2.55
            front=e[0]+v.lr_m*ca.cos(e[2])+.5*v.body_length_m*ca.fabs(ca.cos(e[2]))+.5*v.body_width_m*ca.fabs(ca.sin(e[2]))
            speed=ca.if_else(clear,8.,ca.fmin(nonnegative_sqrt(6*ca.fmax(-2.55-front,0)),8.))
    if name=='intersection':speed=ca.if_else(release_allowed(e,z,v),8.,speed)
    return ca.vertcat(y,heading,speed)


class CommonModel:
    nx=18

    def __init__(self,name,vehicle,config,weight_scale=1.,settings=None):
        if vehicle.dynamics_version!='actuated_v3' or config.horizon!=10:
            raise ValueError('This adapter is declared only for R7 actuated_v3 N10.')
        self.name=name;self.vehicle=vehicle;self.config=config
        self.sat_target=getattr(settings,'sat_target_m',1.5)
        x=ca.SX.sym('x',18);u=ca.SX.sym('u',2);k=ca.SX.sym('k');start=ca.SX.sym('start')
        command=ca.vertcat(x[15]+clip(u[0]-x[15],-.8,.8),u[1])
        ego=step_vehicle(x[:5],command,vehicle)
        sv=[step_vehicle(x[5+5*b:10+5*b],other_action(name,x[5+5*b:10+5*b],x[:5],b,vehicle),vehicle) for b in range(2)]
        xn=ca.vertcat(ego,*sv,command,(ego[3]-x[3])/vehicle.dt_s)
        self.f=ca.Function('next',[x,u],[xn])
        self.fj=ca.Function('next_jac',[x,u],[ca.jacobian(xn,x),ca.jacobian(xn,u)])
        self.r=[];self.h=[];self.derivatives=[]
        weights=np.asarray(config.base_parameters,float)
        axis=np.array([getattr(settings,'lateral_weight',1.),getattr(settings,'heading_weight',1.),1.,getattr(settings,'steering_weight',1.)])
        for branch in range(2):
            z=sv[branch];ref=reference(name,ego,z,branch,vehicle)
            error=ca.vertcat(ego[1]-ref[0],ego[2]-ref[1],ego[3]-ref[2],ego[4])
            residual=ca.vertcat(ca.DM(np.sqrt(vehicle.dt_s*weights[4:8]*weight_scale*axis))*error,
                ca.DM(np.sqrt(vehicle.dt_s*weights[8:10]))*command,
                math.sqrt(vehicle.dt_s*config.jerk_weight)*((xn[17]-x[17])/(5*vehicle.dt_s)),
                ca.if_else(k==config.horizon-1,ca.DM(np.sqrt(weights[:4]*weight_scale*axis*getattr(settings,'terminal_scale',1.)))*error,ca.SX.zeros(4)))
            margin=self.sat_target-(self.sat_target-start)*(.9**(k+1))
            margin=ca.fmax(margin,1.)
            viability=0.
            if name=='intersection' and branch==0:
                front=ego[0]+vehicle.lr_m*ca.cos(ego[2])+.5*vehicle.body_length_m*ca.fabs(ca.cos(ego[2]))+.5*vehicle.body_width_m*ca.fabs(ca.sin(ego[2]))
                viability=ca.if_else(release_allowed(ego,z,vehicle),0.,stop_distance(ego[3],vehicle)-ca.fmax(-2.55-front,0))
            road=.02*ca.logsumexp(-road_gaps(ego,name,vehicle)/.02)+getattr(settings,'road_buffer_m',0.)
            h=ca.vertcat(margin-smooth_clearance(ego,z,vehicle),road,viability,
                u[0]-vehicle.max_accel_mps2,-vehicle.max_brake_mps2-u[0],
                u[1]-vehicle.max_steer_rate_rad_s,-vehicle.max_steer_rate_rad_s-u[1],
                u[0]-x[15]-.8,x[15]-u[0]-.8)
            inputs=[x,u,k,start]
            self.r.append(ca.Function(f'r{branch}',inputs,[residual]))
            self.h.append(ca.Function(f'h{branch}',inputs,[h]))
            xu=ca.vertcat(x,u)
            self.derivatives.append(ca.Function(f'deriv{branch}',inputs,[residual,ca.jacobian(residual,xu),h,ca.jacobian(h,xu)]))
        self.nh=int(h.numel())

    def initial(self,ego,other,previous,acceleration):
        return np.r_[ego,other,other,previous,acceleration].astype(float)

    def start_margin(self,initial):
        return min(self.sat_target,max(1.,float(smooth_clearance(ca.DM(initial[:5]),ca.DM(initial[5:10]),self.vehicle))))

    def condensed(self,prefix):
        N=self.config.horizon
        nodes=prefix+2*(N-prefix)
        controls=ca.MX.sym('controls',nodes*2)
        initial=ca.MX.sym('initial',self.nx);start=ca.MX.sym('start')
        residuals=[];constraints=[]
        for b in range(2):
            x=initial;rs=[];hs=[]
            for k in range(N):
                node=k if k<prefix else prefix+b*(N-prefix)+(k-prefix)
                u=controls[2*node:2*node+2]
                rs.append(self.r[b](x,u,k,start));hs.append(self.h[b](x,u,k,start));x=self.f(x,u)
            residuals.append(ca.vertcat(*rs));constraints.append(ca.vertcat(*hs))
        return ca.Function('condensed',[controls,initial,start],residuals+constraints+[
            ca.jacobian(r,controls) for r in residuals]+[ca.jacobian(h,controls) for h in constraints])
