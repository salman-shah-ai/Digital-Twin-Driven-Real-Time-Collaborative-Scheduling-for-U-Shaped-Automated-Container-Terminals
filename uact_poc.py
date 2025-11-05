
# uact_poc.py
# Simplified U-ACT simulator and compact PPO (PyTorch).
import math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple

Task = namedtuple('Task', ['id','type','remaining_proc','yard_slot'])

class SimpleUACTEnv:
    def __init__(self, n_containers=40, n_agvs=5, seed=0, et_delay_rate=0.1):
        random.seed(seed); np.random.seed(seed)
        self.n_containers = n_containers
        self.n_agvs = n_agvs
        self.et_delay_rate = et_delay_rate
        self.reset()

    def reset(self):
        n_import = self.n_containers // 2
        n_export = self.n_containers - n_import
        self.time = 0.0
        self.done = False
        self.tasks_import_pool = deque()
        self.tasks_export_pool = deque()
        tid = 0
        for i in range(n_import):
            self.tasks_import_pool.append(Task(tid, 'import', remaining_proc=3.0, yard_slot=i))
            tid += 1
        for i in range(n_export):
            self.tasks_export_pool.append(Task(tid, 'export', remaining_proc=3.0, yard_slot=n_import+i))
            tid += 1
        self.agvs = [{'id':i,'status':'idle','available_at':0.0,'load':0} for i in range(self.n_agvs)]
        self.ycs = [{'id':0,'busy_until':0.0,'queue_agv':deque(), 'queue_et':deque(), 'current_task': None}]
        self.et_schedule = []
        base_interval = 5.0
        t = 10.0
        for i in range(len(self.tasks_import_pool)):
            delay = np.random.exponential(scale=base_interval) if random.random() < self.et_delay_rate else np.random.uniform(0, base_interval)
            t += delay
            self.et_schedule.append({'task_id': i, 'arrival': t, 'served': False})
        self.agv_waiting_time = {i:0.0 for i in range(self.n_agvs)}
        self.makespan = 0.0
        self.completed = []
        self.transport_pool = deque()
        for tsk in list(self.tasks_import_pool):
            self.transport_pool.append(tsk)
        for tsk in list(self.tasks_export_pool):
            self.transport_pool.append(tsk)
        return self._get_state()

    def step_physical(self, dt=1.0):
        self.time += dt
        for yc in self.ycs:
            if yc['current_task'] is not None and yc['busy_until'] <= self.time:
                task = yc['current_task']
                yc['current_task'] = None
                self.completed.append((task.id, self.time))
                yc['busy_until'] = 0.0
        for et in self.et_schedule:
            if not et['served'] and et['arrival'] <= self.time:
                yc = self.ycs[0]
                yc['queue_et'].append(et['task_id'])
                et['served'] = True

    def scheduling_point(self):
        idle_agvs = [agv for agv in self.agvs if agv['status']=='idle']
        return len(self.transport_pool)>0 and len(idle_agvs)>0

    def apply_action(self, action_rule_idx):
        task_rule = 'LRTC' if action_rule_idx in [0,2,4] else 'SRTC'
        agv_rule = 'EUTA' if action_rule_idx in [0,1] else ('SPTA' if action_rule_idx in [2,3] else 'NLRA')
        yc = self.ycs[0]
        et_waiting = len(yc['queue_et'])>0
        tasks = list(self.transport_pool)
        if len(tasks)==0:
            return 0
        reverse = True if task_rule=='LRTC' else False
        tasks_sorted = sorted(tasks, key=lambda t: t.remaining_proc, reverse=reverse)
        if et_waiting:
            tasks_sorted = sorted(tasks_sorted, key=lambda t: 0 if t.type=='export' else 1)
        idle_agvs = [agv for agv in self.agvs if agv['status']=='idle']
        if len(idle_agvs)==0:
            return 0
        if agv_rule=='EUTA':
            agv_order = idle_agvs
            random.shuffle(agv_order)
        else:
            agv_order = sorted(idle_agvs, key=lambda a: a['load'])
        assigned = 0
        for agv in agv_order:
            if len(tasks_sorted)==0: break
            task = tasks_sorted.pop(0)
            travel_time = 2.0 + 0.5*task.remaining_proc
            agv['status']='busy'
            agv['available_at'] = self.time + travel_time
            agv['load'] += 1
            assigned += 1
            for i,t in enumerate(self.transport_pool):
                if t.id==task.id:
                    self.transport_pool.remove(t)
                    break
            self.ycs[0]['queue_agv'].append(task.id)
        return assigned

    def advance_to_next_event(self):
        times = []
        for agv in self.agvs:
            if agv['status']=='busy':
                times.append(agv['available_at'])
                if agv['available_at'] <= self.time:
                    agv['status']='idle'
        for et in self.et_schedule:
            if not et['served']:
                times.append(et['arrival'])
        yc = self.ycs[0]
        if yc['current_task'] is None and len(yc['queue_agv'])>0:
            task_id = yc['queue_agv'].popleft()
            yc['current_task'] = Task(task_id, 'unknown', remaining_proc=0.0, yard_slot=0)
            service_time = 1.0 + random.random()*1.0
            yc['busy_until'] = self.time + service_time
            times.append(yc['busy_until'])
        if len(times)==0:
            self.done = True
            return
        next_t = min(times)
        dt = max(0.01, next_t - self.time)
        self.step_physical(dt)

    def get_metrics(self):
        return {'time': self.time, 'completed': len(self.completed)}

    def _get_state(self):
        n_import = sum(1 for t in self.transport_pool if t.type=='import')
        n_export = sum(1 for t in self.transport_pool if t.type=='export')
        imports = [t.remaining_proc for t in self.transport_pool if t.type=='import']
        exports = [t.remaining_proc for t in self.transport_pool if t.type=='export']
        def mean_std(arr):
            if len(arr)==0:
                return 0.0, 0.0
            a = np.array(arr)
            return float(a.mean()), float(a.std())
        cr_i_mean, cr_i_std = mean_std(imports)
        cr_o_mean, cr_o_std = mean_std(exports)
        all_rem = [t.remaining_proc for t in self.transport_pool]
        rem_mean, rem_std = mean_std(all_rem)
        agv_loads = [agv['load'] for agv in self.agvs]
        agv_mean, agv_std = mean_std(agv_loads)
        yc_load = len(self.ycs[0]['queue_agv']) + len(self.ycs[0]['queue_et'])
        yc_mean, yc_std = float(yc_load), 0.0
        qlen_agv = len(self.ycs[0]['queue_agv'])
        qlen_agv_std = 0.0
        agv_wait_mean = np.mean([max(0.0, self.time - agv['available_at']) if agv['status']=='busy' else 0.0 for agv in self.agvs]) if len(self.agvs)>0 else 0.0
        agv_wait_std = 0.0
        qlen_et = len(self.ycs[0]['queue_et'])
        qlen_et_std = 0.0
        et_wait_mean = 0.0
        et_wait_std = 0.0
        state = np.array([
            n_import, n_export,
            cr_i_mean, cr_i_std,
            cr_o_mean, cr_o_std,
            rem_mean, rem_std,
            agv_mean, agv_std,
            yc_mean, yc_std,
            qlen_agv, qlen_agv_std,
            agv_wait_mean, agv_wait_std,
            qlen_et, qlen_et_std,
            et_wait_mean, et_wait_std
        ], dtype=np.float32)
        return state

# PPO components
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64,64)):
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last,h)); layers.append(nn.ReLU()); last = h
        self.shared = nn.Sequential(*layers)
        self.actor = nn.Linear(last, action_dim)
        self.critic = nn.Linear(last, 1)
    def forward(self, x):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value.squeeze(-1)

def entropy_from_logits(logits):
    a = F.softmax(logits, dim=-1)
    return -(a * F.log_softmax(logits, dim=-1)).sum(-1)

def compute_reward(env_before, env_after, alpha=0.6):
    N = env_before.n_containers
    before_completed = len(env_before.completed)
    after_completed = len(env_after.completed)
    delta_cr = (after_completed - before_completed) / max(1, N)
    before_queue = len(env_before.ycs[0]['queue_agv'])
    after_queue = len(env_after.ycs[0]['queue_agv'])
    delta_queue = max(0, after_queue - before_queue)
    reward = alpha * delta_cr - (1-alpha) * (delta_queue / max(1, env_before.n_agvs))
    return reward

def train_one_episode(env, model, optim, max_steps=200, gamma=0.95, clip=0.2):
    states, actions, old_logps, rewards, values = [], [], [], [], []
    s = env.reset()
    total_reward = 0.0
    for step in range(max_steps):
        if env.scheduling_point():
            state = env._get_state()
            logits, value = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            a = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(a)).item()
            env_copy = env
            env.apply_action(a)
            env.advance_to_next_event()
            r = compute_reward(env_copy, env, alpha=0.6)
            states.append(state); actions.append(a); old_logps.append(logp); rewards.append(r); values.append(float(value.item()))
            total_reward += r
        else:
            env.advance_to_next_event()
        if env.done or len(env.completed)>=env.n_containers:
            break
    if len(states)==0:
        return total_reward
    returns = []
    G=0.0
    for r in reversed(rewards):
        G = r + gamma*G
        returns.insert(0,G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean())/(returns.std()+1e-8)
    states_t = torch.tensor(np.array(states), dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)
    old_logps_t = torch.tensor(old_logps, dtype=torch.float32)
    for epoch in range(4):
        logits, values_pred = model(states_t)
        dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
        new_logps = dist.log_prob(actions_t)
        ratio = torch.exp(new_logps - old_logps_t)
        adv = returns - values_pred.detach()
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0-clip, 1.0+clip) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values_pred, returns)
        ent = entropy_from_logits(logits).mean()
        loss = policy_loss + 0.5*value_loss - 0.01*ent
        optim.zero_grad(); loss.backward(); optim.step()
    return total_reward

# Demo run (short)
if __name__ == '__main__':
    state_dim = 20; action_dim = 6
    model = ActorCritic(state_dim, action_dim)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    env = SimpleUACTEnv(n_containers=40, n_agvs=5, seed=2)
    episodes = 20
    for ep in range(episodes):
        r = train_one_episode(env, model, optim, max_steps=300)
        if (ep+1)%5==0:
            print(f'Ep {ep+1}/{episodes}, reward={r:.4f}, time={env.time:.2f}, completed={len(env.completed)}')
    torch.save(model.state_dict(), '/mnt/data/uact_poc_policy.pth')
    print('Saved demo policy to /mnt/data/uact_poc_policy.pth')
