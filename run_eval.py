import argparse, re, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

P1 = """다음은 병원 수술 동의서에 나오는 문장입니다.
의학 지식이 없는 일반 성인 환자가 이 문장을 한 번 읽고 이해할 수 있는지 판단하세요.

문장: {s}

'어려움' 또는 '보통' 중 하나만 출력하세요. 다른 말은 하지 마세요."""

P2 = """당신은 환자 대상 의료 문서의 가독성을 평가하는 전문가입니다.
아래 기준 중 하나라도 해당하면 '어려움', 모두 해당하지 않으면 '보통'으로 판정하세요.

1. 일반인이 모르는 의학 전문용어가 설명 없이 쓰임
2. 문장이 길거나 관형절이 중첩되어 구조 파악이 어려움
3. 서술어가 생략되어 무슨 일이 일어나는지 불명확함
4. 지시어나 접속의 범위가 모호해 두 가지 이상으로 해석됨

문장: {s}

'어려움' 또는 '보통' 중 하나만 출력하세요."""

SHOTS = [("관절경 삽입 부위를 봉합합니다.", "보통"),
         ("견봉 하 공간의 유착된 반흔 조직을 제거한 후 견봉 성형술을 시행합니다.", "어려움"),
         ("위 성공률은 일반적인 성공률이며 임상적인 상황에 따라 차이가 있을 수 있습니다.", "보통"),
         ("문합부유출 : 3% 미만, 재수술 가능", "어려움")]

def make(p, s):
    if p == "P3":
        h = "다음은 병원 수술 동의서 문장의 난이도를 판정한 예시입니다.\n일반 성인 환자가 한 번 읽고 이해할 수 있으면 '보통', 아니면 '어려움'입니다.\n\n"
        return h + "".join(f"문장: {a}\n판정: {b}\n\n" for a, b in SHOTS) + f"문장: {s}\n\n이 문장의 판정만 출력하세요. '어려움' 또는 '보통' 한 단어만."
    return (P1 if p == "P1" else P2).format(s=s)

def parse(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).replace("<think>", "").replace("</think>", "")
    m = re.search(r"(어려움|보통)", t)
    return m.group(1) if m else "PARSE_FAIL"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompt", default="P1", choices=["P1", "P2", "P3"])
ap.add_argument("--csv", default="consent_labeled_draft.csv")
a = ap.parse_args()

df = pd.read_csv(a.csv)
df = df[df["label"].notna()].reset_index(drop=True)
if a.prompt == "P3":
    df = df[~df["sentence"].isin({x for x, _ in SHOTS})].reset_index(drop=True)
print(f"평가 문장 수: {len(df)}")

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()

preds, raws = [], []
for i, r in df.iterrows():
    msgs = [{"role": "user", "content": make(a.prompt, r["sentence"])}]
    try:
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok([txt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    g = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    raws.append(g); preds.append(parse(g))
    print(f"[{i+1}/{len(df)}] {r['id']} 정답={r['label']} 예측={preds[-1]}")

df["pred"] = preds; df["raw_output"] = raws
v = df[df["pred"] != "PARSE_FAIL"]
acc = (v["pred"] == v["label"]).mean() if len(v) else 0
tp = ((v["pred"]=="어려움")&(v["label"]=="어려움")).sum()
fp = ((v["pred"]=="어려움")&(v["label"]=="보통")).sum()
fn = ((v["pred"]=="보통")&(v["label"]=="어려움")).sum()
pr = tp/(tp+fp) if tp+fp else 0
rc = tp/(tp+fn) if tp+fn else 0
f1 = 2*pr*rc/(pr+rc) if pr+rc else 0
print("\n" + "="*40)
print(f"모델: {a.model} | 프롬프트: {a.prompt}")
print(f"파싱실패: {(df['pred']=='PARSE_FAIL').sum()}/{len(df)}")
print(f"정확도: {acc:.3f} (베이스라인 {(df['label']=='어려움').mean():.3f})")
print(f"F1(어려움): {f1:.3f}  P={pr:.3f} R={rc:.3f}")
print("="*40)
n = f"result_{a.model.split('/')[-1]}_{a.prompt}.csv"
df.to_csv(n, index=False); print("저장:", n)
