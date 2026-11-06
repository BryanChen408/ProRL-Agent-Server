---
type: Runbook
title: 固定减1e-4不能补偿BF16归约边界
description: NPU固定反例显示减常量把正确BF16舍入移到错误一侧，同输入归约组织也可跨中点。
tags: [task.debugging, topic.precision]
status: stable
sources: []
created_at: '2026-10-18T08:21:20Z'
updated_at: '2026-10-20T12:33:33Z'
platforms: [a2]
---

# 固定减1e-4不能补偿BF16归约边界

## 现象

旧反向算子在`global_sum*sech^2(gate)`之后、最终BF16 Cast前无条件执行`Adds(...,-1e-4f)`。新最小反例使用FP32 bits0x42ab8001，即85.75000762939453。其直接BF16舍入为86；减1e-4得到FP32值85.74990844726562，再舍入BF16变85.5。原值在BF16相邻值85.5和86的中点85.75上方，固定向下偏置把本来正确的一侧搬到另一侧；上板输出证实此变化。

归约本身也有独立观测：固定输入bits`[0x42ab7fff,0x36000000,0x36000000]`，数值为`[85.74999237060547,2^-19,2^-19]`；FP32 left=(x0+x1)+x2保留0x42ab7fff，right=x0+(x1+x2)得到0x42ab8000（85.75），最终BF16分别85.5和86。此环境torch.sum为left的结果。中点处tie-to-even选择86；精确实数和85.74999618530273仍低于中点。该差异解释为什么很小的FP32归约差异能产生一个BF16量化步长，但不支持无条件加/减常量。

完整机制载体：`probe_group_b.py`的`f32`、`frombits`、`bf16`、`seq`、`sums`函数及`fresh_precision_probe.py`的bias输入/计算/输出单元；上板bits见`npu_observations.json`和`fresh_false.json`。不能以这个新反例声称恢复历史双 RNG 五维搜索，也未验证其4096超块/64lane/8核布局。没有修改参考、容差或benchmark输入来使候选通过。

## 适用场景

- 工程形态：Ascend NPU算子精度校准的小规模框架探针。
- API路径：FP32归约与最终BF16 Cast之前的固定Adds(-1e-4)。
- 验证平台与版本：a2/Ascend910B4，CANN9.1.0，torch/torch_npu2.9.0。
- 适用粒度：`operator_family`。
- 路线优先级：`banned`。
- 不适用或未验证场景：不恢复旧双RNG搜索或任何特定归约布局；反例不宣称整个旧算子已修复。

## 判别方法

以下为实际执行公共模块 `probe_group_b.py` 的全文，包含导入、闭包上下文、输入生成、接口调用和原始输出记录；依赖该模块的 fresh 进程脚本另列如下。所有代码保持执行版本逐字一致。

```python
#!/usr/bin/env python3
"""Small, deterministic calibration only. No benchmark, model fitting or submission.
CPU mode has no dependencies. NPU mode requires the operator's installed torch_npu.
Every tensor is copied back and archived as raw hex, values and sha256.
"""
import argparse, hashlib, json, math, os, platform, struct, subprocess, sys, time
from pathlib import Path

def f32(x):
    try:return struct.unpack('<f',struct.pack('<f',x))[0]
    except OverflowError:return math.copysign(float('inf'),x)
def f16(x):
    try:return struct.unpack('<e',struct.pack('<e',x))[0]
    except OverflowError:return math.copysign(float('inf'),x)
def frombits(b):return struct.unpack('<f',struct.pack('<I',b))[0]
def bits(x):return '%08x'%struct.unpack('<I',struct.pack('<f',x))[0]
def rne11(x):
    u=int(bits(x),16)
    if (u&0x7f800000)==0x7f800000:return x
    return frombits((u+0x7ff+((u>>12)&1))&0xfffff000)
def bf16(x):
    u=int(bits(x),16)
    return frombits((u+0x7fff+((u>>16)&1))&0xffff0000)
def veltkamp(x):
    c=f32(f32(x)*4097.0);return f32(c-f32(c-x))
def seq(xs):
    r=0.0
    for x in xs:r=f32(r+x)
    return r

def iou(a,b,rounder):
    w=rounder(max(0.,rounder(min(a[2],b[2])-max(a[0],b[0]))))
    h=rounder(max(0.,rounder(min(a[3],b[3])-max(a[1],b[1]))))
    inter=rounder(w*h)
    ar=[rounder(rounder(x[2]-x[0])*rounder(x[3]-x[1])) for x in [a,b]]
    union=rounder(rounder(ar[0]+ar[1])-inter)
    return {'width':w,'height':h,'inter':inter,'areas':ar,'union':union,'iou':rounder(inter/union)}

def find_nms():
    # All coordinates are exactly representable half inputs before simulating stages.
    import random
    rng=random.Random(20261018)
    for _ in range(20000):
        a=[0.,0.,f16(rng.uniform(.5,3)),f16(rng.uniform(.5,3))]
        b=[f16(rng.uniform(0,.4)),f16(rng.uniform(0,.4)),f16(rng.uniform(.5,3)),f16(rng.uniform(.5,3))]
        v32=iou(a,b,f32);v16=iou(a,b,f16);last=f16(v32['iou'])
        vals=sorted(set([v32['iou'],v16['iou'],last]))
        if len(vals)==3 and vals[0]>.05:
            thresholds=[f32((x+y)/2) for x,y in zip(vals,vals[1:])]
            if len(set(thresholds))==2:return {'boxes':[a,b],'fp32_stages':v32,'fp16_stages':v16,'fp32_final16':last,'thresholds':thresholds}
    raise RuntimeError('No separating pair')

def cpu():
    vals=[frombits(x) for x in [0x3f800800,0x3f801800,0x00000001,0x000116c2,0x79800000,0x7b4097ce]]
    rows=[{'input':x,'input_fp32_bits':bits(x),'rne11':rne11(x),'rne11_bits':bits(rne11(x)),'veltkamp':veltkamp(x),'veltkamp_bits':bits(veltkamp(x))} for x in vals]
    xs=[frombits(0x42ab7fff),2.**-19,2.**-19]
    left=seq(xs);right=f32(xs[0]+f32(xs[1]+xs[2]))
    bias=[frombits(0x42ab8001),frombits(0xc2ab8001)]
    return {'C056_veltkamp_counterexamples':rows,'C059_midpoint':{'inputs':xs,'input_fp32_bits':[bits(x) for x in xs],'left':left,'left_bits':bits(left),'right':right,'right_bits':bits(right),'left_bf16':bf16(left),'right_bf16':bf16(right),'exact_sum':sum(xs)},'C059_fixed_bias_counterexamples':[{'input':x,'unbiased_bf16':bf16(x),'subtract_1e_4_bf16':bf16(f32(x-1e-4))} for x in bias],'C044_separating_pair':find_nms(),'claim_scope':'CPU arithmetic simulations only; no AscendC or reference-internal DAG inferred.'}

def record(t):
    t=t.detach().cpu().contiguous()
    raw=bytes(t.reshape(-1).view(torch.uint8).tolist())
    return {'dtype':str(t.dtype),'shape':list(t.shape),'stride':list(t.stride()),'raw_hex':raw.hex(),'sha256':hashlib.sha256(raw).hexdigest(),'values':t.reshape(-1).tolist()}

def read_setting(obj,name):
    try:return getattr(obj,name)
    except Exception as e:return {'unavailable':str(e)}
def npu_probe(device):
    global torch
    import torch
    import torch_npu
    torch.npu.set_device(device)
    env={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torch_npu':getattr(torch_npu,'__version__','unknown'),'device':device,'device_name':torch.npu.get_device_name(device),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'precision_environment':{k:v for k,v in os.environ.items() if k.startswith(('ASCEND','ACL_OP','NPU_')) and not any(s in k for s in ['TOKEN','SECRET','PASSWORD'])},'settings':{}}
    for nm,obj in [('torch.npu.conv',getattr(torch.npu,'conv',None)),('torch.npu.matmul',getattr(torch.npu,'matmul',None)),('torch.backends.cuda.matmul',torch.backends.cuda.matmul)]:
        env['settings'][nm]={x:read_setting(obj,x) for x in ['allow_hf32','allow_tf32']}
    for command in [['npu-smi','info']]:
        try:r=subprocess.run(command,capture_output=True,text=True,timeout=10);env[' '.join(command)]={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        except Exception as e:env[' '.join(command)]={'error':str(e)}
    out={'environment':env,'cpu_simulations':cpu(),'observations':{},'limitations':['Framework outputs do not establish AscendC Compare/Rsqrt semantics or internal reduction/multiply DAG.','No candidate AIV-vs-Cube performance claim; no full benchmark.']}
    # Each group isolated so an unavailable API never becomes an invented observation.
    def run(key,fn):
        try:out['observations'][key]=fn();torch.npu.synchronize()
        except Exception as e:out['observations'][key]={'error':type(e).__name__+': '+str(e)}
    def nms():
        p=out['cpu_simulations']['C044_separating_pair'];rows=[]
        for dtype in [torch.float16,torch.float32]:
            boxes=torch.tensor(p['boxes'],dtype=dtype,device=device);scores=torch.tensor([.9,.8],dtype=dtype,device=device)
            for threshold in p['thresholds']+[p['fp32_final16']]:
                th=torch.tensor(threshold,dtype=torch.float32,device=device);st=torch.tensor(-1.,dtype=torch.float32,device=device)
                r=torch_npu.npu_nms_v4(boxes,scores,2,th,st,pad_to_max_output_size=True)
                rows.append({'boxes':record(boxes),'scores':record(scores),'threshold':record(th),'result':[record(x) for x in r],'candidate_suppression_gt':{'fp32':p['fp32_stages']['iou']>threshold,'step_fp16':p['fp16_stages']['iou']>threshold,'final_fp16':p['fp32_final16']>threshold}})
        return rows
    run('C044_actual_npu_nms_v4',nms)
    def conv_and_matmul():
        rows=[]
        # One active channel prevents reduction-order explanations of product differences.
        vals=[frombits(x) for x in [0x3f800801,0x3f801801,0x3f802801,0xbf800801,0x3f7ff801,0x3f800001,0x3f801fff,0x3f802001]]
        x=torch.zeros((1,16,8),dtype=torch.float32);x[0,0,:]=torch.tensor(vals)
        w=torch.zeros((16,16,1),dtype=torch.float32);w[:,0,0]=frombits(0x3f801801)
        A=torch.tensor(vals,dtype=torch.float32).reshape(8,1).repeat(1,16);A[:,1:]=0
        B=torch.zeros((16,16),dtype=torch.float32);B[:,0]=frombits(0x3f801801)
        for allow in [None,False,True]:
            changed=[]
            try:
                if allow is not None:
                    for obj in [getattr(torch.npu,'conv',None),getattr(torch.npu,'matmul',None)]:
                        if obj is not None and hasattr(obj,'allow_hf32'):
                            old=obj.allow_hf32;obj.allow_hf32=allow;changed.append((obj,old))
                xd=x.to(device);wd=w.to(device);ad=A.to(device);bd=B.to(device)
                conv=torch.nn.functional.conv1d(xd,wd);mm=torch.matmul(ad,bd.T)
                rows.append({'allow_hf32_requested':allow,'switches_changed':len(changed),'x':record(x),'weight':record(w),'conv1d':record(conv),'A':record(A),'B':record(B),'matmul_ABt':record(mm),'sim_full_f32_product':[f32(v*frombits(0x3f801801)) for v in vals],'sim_rne11_product':[f32(rne11(v)*rne11(frombits(0x3f801801))) for v in vals]})
            finally:
                for obj,old in changed:obj.allow_hf32=old
        return rows
    run('C056_C078_single_active_channel',conv_and_matmul)
    def sums():
        rows=[]
        for xs in [[frombits(0x42ab7fff),2.**-19,2.**-19],[1e8,1.,-1e8]]:
            t=torch.tensor(xs,dtype=torch.float32,device=device)
            outputs={'torch_sum':t.sum(),'left':(t[0]+t[1])+t[2],'right':t[0]+(t[1]+t[2])}
            rows.append({'input':record(t),'outputs':{k:record(v) for k,v in outputs.items()},'bf16_outputs':{k:record(v.to(torch.bfloat16)) for k,v in outputs.items()}})
        return rows
    run('C059_sum_midpoint',sums)
    def rsqrts():
        t=torch.tensor([frombits(x) for x in [0x3f800000,0x40400000,0x3727c5ac,0x2edbe6ff,0x00800000,0x00000001,0x7f7fffff,0x00000000,0x80000000,0x7f800000,0xbf800000,0x7fc12345]],dtype=torch.float32,device=device)
        y=torch.rsqrt(t);z=torch.reciprocal(torch.sqrt(t))
        return {'fixed_var_plus_eps':record(t),'framework_rsqrt':record(y),'framework_sqrt_reciprocal':record(z),'fp64_math_reference':[1/math.sqrt(x) if x>0 and math.isfinite(x) else ('positive_inf' if x==0 else (0. if x==float('inf') else 'nan')) for x in t.cpu().tolist()]}
    run('C066_framework_only_rsqrt',rsqrts)
    env['loaded_ascend_libraries'] = sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if any(s in line.lower() for s in ['ascend','torch_npu','libacl'])))
    env['ascend_version_files'] = {str(p): p.read_text(errors='replace') for p in [Path('/usr/local/Ascend/ascend-toolkit/latest/version.cfg'),Path('/usr/local/Ascend/cann-9.1.0/version.cfg'),Path('/usr/local/Ascend/cann-9.1.0/version.info')] if p.is_file()}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['cpu','npu'],required=True);ap.add_argument('--device',default='npu:1');ap.add_argument('--output',required=True);args=ap.parse_args()
    p=Path(args.output)
    if p.exists():raise SystemExit('Refuse to overwrite '+str(p))
    out=cpu() if args.mode=='cpu' else npu_probe(args.device)
    out['producer']={'command':sys.argv,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=True)+'\n');print(json.dumps({'output':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'groups':list(out.get('observations',out))},ensure_ascii=False))
if __name__=='__main__':main()
```

以下为实际fresh进程脚本全文；它从上述公共模块导入函数，在首次算子前设置模式，并保存读回与原始输出。

```python
#!/usr/bin/env python3
"""One mode per fresh process, set before first tensor op; no benchmark."""
import hashlib,inspect,json,sys
from pathlib import Path
import torch,torch_npu
from probe_group_b import frombits,record,f32,rne11
import probe_group_b
probe_group_b.torch=torch
mode=sys.argv[1]=='true';out=Path(sys.argv[2]);torch.npu.set_device(0)
if out.exists():raise RuntimeError('Refuse overwrite')
torch.npu.conv.allow_hf32=mode;torch.npu.matmul.allow_hf32=mode
settings={'conv_readback':torch.npu.conv.allow_hf32,'matmul_readback':torch.npu.matmul.allow_hf32}
vals=[frombits(x) for x in [0x3f800801,0x3f801801,0x3f802801,0xbf800801,0x3f7ff801,0x3f800001,0x3f801fff,0x3f802001]]
x=torch.zeros((1,16,8),dtype=torch.float32);x[0,0,:]=torch.tensor(vals)
w=torch.zeros((16,16,1),dtype=torch.float32);w[:,0,0]=frombits(0x3f801801)
a=torch.zeros((8,16),dtype=torch.float32);a[:,0]=torch.tensor(vals)
b=torch.zeros((16,16),dtype=torch.float32);b[:,0]=frombits(0x3f801801)
y=torch.nn.functional.conv1d(x.npu(),w.npu());mm=torch.matmul(a.npu(),b.npu().T)
bias=torch.tensor([frombits(0x42ab8001)],dtype=torch.float32,device='npu:0');biased=bias-1e-4
v=torch.tensor([frombits(i) for i in [0x3f800800,0x3f801800,0x00000001,0x000116c2,0x79800000]],dtype=torch.float32,device='npu:0');c=v*4097.;t=c-v;hi=c-t
src={}
for obj in [torch.npu.conv,torch.npu.matmul]:
 try:p=inspect.getsourcefile(type(obj))
 except (TypeError,OSError):p=getattr(obj,'__file__',None)
 if p and Path(p).is_file():src[p]={'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest(),'text':Path(p).read_text()}
d={'mode_requested':mode,'settings_before_ops':settings,'conv_output':record(y),'matmul_output':record(mm),'x':record(x),'weight':record(w),'A':record(a),'B':record(b),'bias_input':record(bias),'unbiased_bf16':record(bias.to(torch.bfloat16)),'biased_fp32':record(biased),'biased_bf16':record(biased.to(torch.bfloat16)),'veltkamp_input':record(v),'veltkamp_framework_output':record(hi),'settings_after_ops':{'conv_readback':torch.npu.conv.allow_hf32,'matmul_readback':torch.npu.matmul.allow_hf32},'installed_settings_sources':src,'loaded_ascend_libraries':sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/Ascend/' in line)),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=True)+'\n');print(json.dumps({'output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'mode':mode,'conv_first8':y.detach().cpu().flatten()[:8].tolist(),'matmul_firstcol':mm.detach().cpu()[:,0].tolist()}))
```

实际执行命令（容器已source实际CANN9.1.0环境，local device0）：

```sh
python3 probe_group_b.py --mode npu --device npu:0 --output npu_observations.json
python3 fresh_precision_probe.py false fresh_false.json
python3 fresh_precision_probe.py true fresh_true.json
```

## 设计决策

选择冻结bits和小输入以区分精度路径，而不是从张量dtype或历史全任务通过反推内部DAG。保留未支持的解释，不把新来源称为旧会话恢复。

## 系统信息

本次是固定输入的小规模框架参考校准，使用Ascend910B4（a2）、公开结果所记CANN9.1.0版本、torch/torch_npu2.9.0。源码载体为`probe_group_b.py`和`fresh_precision_probe.py`，原始结果记录输入/输出dtype、shape、stride、raw hex及SHA256。没有运行整benchmark或官网提交，也未加载旧NMS扩展。

## 验证信号

以下来自本次实际小探针的原始机器结果，而非旧摘要。
- 固定输入85.75000762939453直接BF16=86；减1e-4后BF16=85.5。（来源平台：a2；证据等级：tool_observed）
- 固定三值的left/right FP32结果分别0x42ab7fff/0x42ab8000，最终BF16=85.5/86；本环境torch.sum为left。（来源平台：a2；证据等级：tool_observed）

## 失效边界

不恢复旧双RNG搜索或任何特定归约布局；反例不宣称整个旧算子已修复。其他平台未验证，推断可迁移须重新校准。

## 坏实践

将一次数据类型声明或历史通过总结当作未观测内部数值路径的证明；为对齐少数边界案例采用无条件补偿而不留反例。

## 性能影响

未测性能，不作性能排序或收益声称。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/framework/evidence/reference.json#/observations/C059_sum_midpoint`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/framework/evidence/fresh_false.json#/bias_input`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/framework/evidence/fresh_false.json#/unbiased_bf16`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/framework/evidence/fresh_false.json#/biased_fp32`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/framework/evidence/fresh_false.json#/biased_bf16`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/framework/probe_group_b.py`；`cannbot_non_level4_split_review_20261019/evidence_public/dynamic/framework/fresh_precision_probe.py`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。 公开dynamic投影保留数值、设备名、源码/模块或脚本身份及9.1.0版本标记；不包含完整同PID runtime库路径/hash，不能据此独立认证全部加载库。
