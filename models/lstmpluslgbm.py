# -*- coding: utf-8 -*-
"""
Copy of [앙상블3]_LSTM을 활용한 메뉴별 매출 수량 예측.ipynb
LSTM + LGBM Residual Hybrid
"""

# =========================
# LSTM + LGBM Residual Hybrid (rule-safe)
# =========================

import os, math, random, platform, glob
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
from google.colab import drive
drive.mount('/content/drive')

# --- try LightGBM (fallback=sklearn GBR) ---
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_LGBM = False

# -------------------------
# Seed & device
# -------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type=="cuda":
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

print("Python:", platform.python_version())
print("Torch  :", torch.__version__)
print("CUDA   :", torch.version.cuda)
print("Device :", DEVICE)

# -------------------------
# Paths & Const
# -------------------------
BASE_DIR   = "/content/drive/MyDrive/lgaimers"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_DIR   = f"{BASE_DIR}/"
SAMPLE_SUB = f"{BASE_DIR}/sample_submission.csv"
OUT_PATH   = f"{BASE_DIR}/submission.csv"

LOOKBACK, PREDICT = 28, 7
EPOCHS    = 10
BATCH     = 512
LR_MAX    = 2e-3
HIDDEN    = 160
LAYERS    = 2
DROPOUT   = 0.25
SCALE_MODE = "log1p"
USE_RECENCY_WEIGHTS = True
NEGATIVE_STRATEGY_TRAIN = "zero"
PRIORITY_SHOPS = {"담하", "미라시아"}
PRIORITY_W = 1.0                     # 담하/미라시아 업장 추가 가중치
GBM_FEATURES = [
    "shop_id","type_id","fut_w","fut_m",
    "last_s","mean7_s","mean28_s","trend_s","log_lstm"
]
USE_RESIDUAL = True

"""# 유틸/전처리

"""

# -------------------------
# Utils / feature helpers
# -------------------------
def split_shop_menu(name):
    if isinstance(name, str) and "_" in name:
        a,b = name.split("_", 1); return a,b
    return name,""

def categorize_menu(menu_name: str) -> str:
    if not isinstance(menu_name, str): return "기타"
    s = menu_name.lower()
    summer = ["빙수","shaved","아이스","냉","콜드","여름","수박","망고"]
    winter = ["따뜻","hot","온","겨울","호빵","어묵","오뎅","호떡","붕어빵"]
    alcohol = ["맥주","soju","소주","와인","wine","막걸리","사케","위스키","칵테일","highball","하이볼","소맥","주류","beer","sake"]
    nonalc = ["주스","에이드","스무디","라떼","커피","차","tea","티","아메리카노","콜라","사이다","스프라이트","음료","우유"]
    venue  = ["대관","장소","홀대여","룸대여","room","space","연회장","banquet","대여","컨벤션","hall","convention"]
    group  = ["단체","뷔페","코스","패키지","세트","연회","행사"]
    side   = ["사이드","안주","튀김","감자","김치","디저트","케이크","빵","쿠키","과자","토핑"]
    main   = ["정식","세트","메인","bbq","회","스테이크","파스타","라면","국수","비빔밥","한정식","덮밥","초밥","피자","컵밥","돈가스","짜장","짬뽕","불고기","갈비","탕","찌개","국","면","밥","버거","샐러드"]
    if any(k in s for k in summer): return "계절_여름"
    if any(k in s for k in winter): return "계절_겨울"
    if any(k in s for k in venue):  return "장소대여"
    if any(k in s for k in alcohol):return "음료_술"
    if any(k in s for k in nonalc): return "음료_무알코올"
    if any(k in s for k in group):  return "음식_단체"
    if any(k in s for k in side):   return "음식_사이드"
    if any(k in s for k in main):   return "음식_메인"
    return "기타"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["영업일자"] = pd.to_datetime(df["영업일자"], errors="coerce")
    df["요일"] = df["영업일자"].dt.weekday.astype(int).clip(0,6)
    df["개월"] = (df["영업일자"].dt.month-1).astype(int).clip(0,11)
    shops, menus = zip(*df["영업장명_메뉴명"].map(split_shop_menu))
    df["업장"] = list(shops)
    df["메뉴명"] = list(menus)
    df["메뉴유형"] = df["메뉴명"].map(categorize_menu)
    return df

def handle_negatives_train(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    df = df.copy()
    if strategy == "zero":
        df.loc[df["매출수량"]<0,"매출수량"]=0
    elif strategy == "drop":
        df = df[df["매출수량"]>=0].reset_index(drop=True)
    elif strategy == "abs":
        df["매출수량"]=df["매출수량"].abs()
    elif strategy == "neighbor_avg":
        def repair(g):
            s = g["매출수량"].astype(float); mask = s<0
            anchor = s.mask(mask); prev, nxt = anchor.ffill(), anchor.bfill()
            rep = s.copy(); rep[mask] = ((prev+nxt)/2.0)[mask]
            g = g.copy(); g["매출수량"] = rep.fillna(prev).fillna(nxt).fillna(0.0)
            return g
        df = df.groupby("영업장명_메뉴명", group_keys=False).apply(repair).reset_index(drop=True)
    return df

class IDEncoder:
    def __init__(self, use_unk=True):
        self.map = {"<UNK>":0} if use_unk else {}
        self.use_unk=use_unk
    def fit(self, arr):
        for v in arr:
            if v not in self.map: self.map[v]=len(self.map)
        return self
    def transform(self, arr):
        return np.array([self.map.get(v,0) for v in arr], dtype=np.int64)
    @property
    def size(self): return len(self.map)
    def unk_index(self): return 0

def smape_torch(y_true, y_pred, eps=1e-6):
    denom = y_true.abs() + y_pred.abs() + eps
    mask  = (y_true.abs() > 0).float()
    term  = 2.0 * (y_pred - y_true).abs() / denom
    per_sample = (term * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return per_sample  # [B]

# ----- Scalers -----
class Log1pScaler:
    def fit(self, x): return self
    def transform(self, x): return np.log1p(np.clip(np.asarray(x, float), 0, None))
    def inverse_transform(self, x): return np.expm1(np.asarray(x, float))

def make_scaler(tr_df, mode="log1p"):
    if mode=="robust":
        sc = RobustScaler().fit(tr_df[["매출수량"]])
        class _Wrap:
            def __init__(self, sc): self.sc=sc
            def fit(self, x): return self
            def transform(self, x):
                return self.sc.transform(np.asarray(x, float).reshape(-1,1)).reshape(-1)
            def inverse_transform(self, x):
                return self.sc.inverse_transform(np.asarray(x, float).reshape(-1,1)).reshape(-1)
        return _Wrap(sc)
    else:
        return Log1pScaler()

from pandas.api.types import CategoricalDtype

CAT_COLS = ["shop_id","type_id","fut_w","fut_m"]
NUM_COLS = ["last_s","mean7_s","mean28_s","trend_s","log_lstm"]

def cast_gbm_dtypes(df, shop_enc, type_enc):
    df = df.copy()
    cat_types = {
        "shop_id": CategoricalDtype(categories=list(range(shop_enc.size))),
        "type_id": CategoricalDtype(categories=list(range(type_enc.size))),
        "fut_w":   CategoricalDtype(categories=list(range(7))),
        "fut_m":   CategoricalDtype(categories=list(range(12))),
    }
    # 카테고리 컬럼: 반드시 동일 카테고리 집합으로 캐스팅
    for c in CAT_COLS:
        df[c] = df[c].astype("int32").astype(cat_types[c])
    # 수치 컬럼: float32로 고정
    for c in NUM_COLS:
        df[c] = df[c].astype(np.float32)
    # 열 순서도 고정
    return df[CAT_COLS + NUM_COLS]

"""# Dataset & Model 정의"""

# -------------------------
# Torch dataset/model
# -------------------------
class SalesDatasetTrain(Dataset):
    """
    train.csv 전체에서 (28 -> 7) 윈도우 생성.
    sample_weight = 최근성^t * (담하/미라시아 가중)
    """
    def __init__(self, df, scaler, shop_enc, type_enc, lookback=28, predict=7, recency_gamma=0.997):
        self.samples=[]; self.weights=[]
        df = df.sort_values(["영업장명_메뉴명","영업일자"]).copy()
        for (series, shop, mtype), g in df.groupby(["영업장명_메뉴명","업장","메뉴유형"]):
            g = g.sort_values("영업일자").reset_index(drop=True)
            n = len(g) - (lookback+predict) + 1
            if n<=0: continue
            vals = scaler.transform(g["매출수량"].values)
            wd   = g["요일"].values.astype(int)
            mo   = g["개월"].values.astype(int)
            sid  = shop_enc.map.get(shop, shop_enc.unk_index())
            tid  = type_enc.map.get(mtype, type_enc.unk_index())
            pri_w = PRIORITY_W if shop in PRIORITY_SHOPS else 1.0
            for i in range(n):
                past = slice(i, i+lookback)
                fut  = slice(i+lookback, i+lookback+predict)
                x  = vals[past].astype(np.float32)
                y  = vals[fut].astype(np.float32)
                w_p= wd[past]; m_p= mo[past]
                w_f= wd[fut];  m_f= mo[fut]
                last = x[-1]; mean7 = x[-7:].mean(); mean28 = x.mean()
                stats = np.array([last, mean7, mean28], np.float32)
                self.samples.append((
                    torch.tensor(x).unsqueeze(-1),                # [28,1]
                    torch.tensor(w_p, dtype=torch.long),          # [28]
                    torch.tensor(m_p, dtype=torch.long),          # [28]
                    torch.tensor(w_f, dtype=torch.long),          # [7]
                    torch.tensor(m_f, dtype=torch.long),          # [7]
                    torch.tensor(int(sid), dtype=torch.long),
                    torch.tensor(int(tid), dtype=torch.long),
                    torch.tensor(stats),                          # [3]
                    torch.tensor(y),                              # [7]
                    torch.tensor(1.0)                             # placeholder (filled below)
                ))
                age = (n-1) - i
                self.weights.append( (recency_gamma**age) * pri_w )
        self.weights = np.array(self.weights, np.float32)
        # fill weights into tuples
        for k in range(len(self.samples)):
            tup = list(self.samples[k]); tup[-1] = torch.tensor(self.weights[k], dtype=torch.float32)
            self.samples[k] = tuple(tup)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
class LSTMCalHead(nn.Module):
    def __init__(self, hidden=128, layers=2, n_shops=1, n_types=1, dropout=0.3, stats_dim=3):
        super().__init__()
        self.emb_w   = nn.Embedding(7, 4)
        self.emb_m   = nn.Embedding(12,3)
        self.emb_shop= nn.Embedding(max(1,n_shops), 16)
        self.emb_type= nn.Embedding(max(1,n_types), 8)
        self.lstm = nn.LSTM(1+4+3, hidden, layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        in_dim = hidden*2 + 16 + 8 + 4 + 3 + stats_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 1)
        )
    def forward(self, x_val, w_p, m_p, w_f, m_f, sid, tid, stats):
        ew = self.emb_w(w_p); em = self.emb_m(m_p)
        x  = torch.cat([x_val, ew, em], dim=2)   # [B,28,1+4+3]
        enc,_ = self.lstm(x)                     # [B,28,2H]
        enc = enc[:, -1, :]                      # [B,2H]
        es, et = self.emb_shop(sid), self.emb_type(tid)
        outs=[]
        for t in range(w_f.shape[1]):            # 7 steps
            ewf = self.emb_w(w_f[:,t]); emf = self.emb_m(m_f[:,t])
            h = torch.cat([enc, es, et, ewf, emf, stats], dim=1)
            y = self.head(self.drop(h))          # [B,1]
            outs.append(y)
        return torch.cat(outs, dim=1)            # [B,7]

"""# Train LSTM

"""

# -------------------------
# Train LSTM (Stage A)
# -------------------------
def train_and_build_lstm(tr_path):
    tr = pd.read_csv(tr_path)
    tr = add_features(handle_negatives_train(tr, NEGATIVE_STRATEGY_TRAIN))

    shop_enc = IDEncoder().fit(tr["업장"].tolist())
    type_enc = IDEncoder().fit(tr["메뉴유형"].tolist())
    scaler = make_scaler(tr, mode=SCALE_MODE)

    train_ds = SalesDatasetTrain(tr, scaler, shop_enc, type_enc, LOOKBACK, PREDICT, recency_gamma=0.997)
    if len(train_ds)==0: raise RuntimeError("No training samples.")

    if USE_RECENCY_WEIGHTS:
        weights = train_ds.weights / (train_ds.weights.mean() + 1e-8)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                                  num_workers=2, pin_memory=(DEVICE.type=="cuda"))
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                  num_workers=2, pin_memory=(DEVICE.type=="cuda"))

    model = LSTMCalHead(HIDDEN, LAYERS, shop_enc.size, type_enc.size, DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR_MAX, steps_per_epoch=len(train_loader), epochs=EPOCHS,
        pct_start=0.30, div_factor=10.0, final_div_factor=1e3, anneal_strategy="cos"
    )
    l1 = nn.L1Loss(reduction="none")   # per-sample

    for ep in range(1, EPOCHS+1):
        model.train(); run=0.0
        pbar = tqdm(train_loader, desc=f"[LSTM] epoch {ep}/{EPOCHS}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            x, w_p, m_p, w_f, m_f, sid, tid, stats, y, wgt = [t.to(DEVICE) for t in batch]
            opt.zero_grad(set_to_none=True)
            pred = model(x, w_p, m_p, w_f, m_f, sid, tid, stats)          # scaled
            if SCALE_MODE == "log1p":
                pred_orig = torch.expm1(pred); y_orig = torch.expm1(y)
            else:
                raise NotImplementedError("Use log1p mode for this hybrid script.")
            smape_vec = smape_torch(y_orig, pred_orig)     # [B]
            l1_vec    = l1(pred, y).mean(dim=1)            # [B]
            loss_vec  = 0.6*l1_vec + 0.4*smape_vec         # [B]
            wnorm = wgt / (wgt.mean() + 1e-8)
            loss = (loss_vec * wnorm).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); scheduler.step()
            run += float(loss.item())
            pbar.set_postfix(loss=f"{run/step:.4f}")

    return model, scaler, shop_enc, type_enc

"""# LGBM

"""

# -------------------------
# Stage B: LGBM on LOG-residuals (stable)
# -------------------------
RESID_BETA = 0.25               # 잔차 반영 강도(0~1). 0.5 권장
DELTA_CLIP = 0.35          # 로그-잔차 클립 범위(±). 0.7이면 약 ×0.5~×2.0 범위

def _window_rows_for_lgbm(tr_df, model, scaler, shop_enc, type_enc):
    """
    각 (28->7) 윈도우에서
      features: scaled stats + 미래 요일/월 + shop/type id + log_lstm_pred
      target  : delta = log1p(y_true) - log1p(y_lstm)
      weight  : 최근성 * 우선업장
    로 테이블 생성
    """
    rows = []
    df = tr_df.sort_values(["영업장명_메뉴명","영업일자"]).copy()

    for (series, shop, mtype), g in df.groupby(["영업장명_메뉴명","업장","메뉴유형"]):
        g = g.sort_values("영업일자").reset_index(drop=True)
        n = len(g) - (LOOKBACK + PREDICT) + 1
        if n <= 0: continue

        vals_orig   = g["매출수량"].values.astype(float)
        vals_scaled = scaler.transform(vals_orig)
        wd = g["요일"].values.astype(int)
        mo = g["개월"].values.astype(int)

        sid = shop_enc.map.get(shop, shop_enc.unk_index())
        tid = type_enc.map.get(mtype, type_enc.unk_index())
        pri_w = PRIORITY_W if shop in PRIORITY_SHOPS else 1.0

        for i in range(n):
            past = slice(i, i+LOOKBACK)
            fut  = slice(i+LOOKBACK, i+LOOKBACK+PREDICT)
            x_s = vals_scaled[past].astype(np.float32)
            y_o = vals_orig[fut].astype(np.float32)
            w_p, m_p = wd[past], mo[past]
            w_f, m_f = wd[fut],  mo[fut]

            # LSTM 예측 (원단위 → 로그로 변환)
            with torch.no_grad():
                x_t   = torch.tensor(x_s).unsqueeze(0).unsqueeze(-1).to(DEVICE)
                w_p_t = torch.tensor(w_p, dtype=torch.long).unsqueeze(0).to(DEVICE)
                m_p_t = torch.tensor(m_p, dtype=torch.long).unsqueeze(0).to(DEVICE)
                w_f_t = torch.tensor(w_f, dtype=torch.long).unsqueeze(0).to(DEVICE)
                m_f_t = torch.tensor(m_f, dtype=torch.long).unsqueeze(0).to(DEVICE)
                sid_t = torch.tensor([sid], dtype=torch.long).to(DEVICE)
                tid_t = torch.tensor([tid], dtype=torch.long).to(DEVICE)
                stats = np.array([x_s[-1], x_s[-7:].mean(), x_s.mean()], np.float32)
                st_t  = torch.tensor(stats).unsqueeze(0).to(DEVICE)

                model.eval()
                pred_scaled = model(x_t, w_p_t, m_p_t, w_f_t, m_f_t, sid_t, tid_t, st_t).cpu().numpy().reshape(-1)
                y_lstm = np.expm1(pred_scaled)

            last = x_s[-1]; mean7 = x_s[-7:].mean(); mean28 = x_s.mean()
            trend = last - mean7

            age = (n-1) - i
            samp_w = (0.997**age) * pri_w

            # target: 로그-잔차
            log_true = np.log1p(y_o)
            log_lstm = np.log1p(y_lstm)
            delta    = (log_true - log_lstm)  # 길이 7

            for t in range(PREDICT):
                rows.append({
                    "h": t+1,
                    "shop_id": sid, "type_id": tid,
                    "fut_w": int(w_f[t]), "fut_m": int(m_f[t]),
                    "last_s": float(last), "mean7_s": float(mean7), "mean28_s": float(mean28),
                    "trend_s": float(trend),
                    "log_lstm": float(log_lstm[t]),
                    "delta": float(delta[t]),
                    "weight": float(samp_w),
                })
    return pd.DataFrame(rows)

def train_lgbm_residual(tr_df, model, scaler, shop_enc, type_enc):
    table = _window_rows_for_lgbm(tr_df, model, scaler, shop_enc, type_enc)
    models = {}
    for h in range(1, PREDICT+1):
        sub = table[table["h"]==h].copy()
        X = sub[GBM_FEATURES].copy()
        y = sub["delta"].astype(np.float32)
        w = sub["weight"].astype(np.float32)

        # 여기서 dtype/카테고리 고정 적용
        X = cast_gbm_dtypes(X, shop_enc, type_enc)

        if HAS_LGBM:
            m = lgb.LGBMRegressor(
                objective="l2",
                n_estimators=300, learning_rate=0.05,
                num_leaves=31, min_child_samples=64,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=2.0, random_state=SEED
            )
            m.fit(
                X, y, sample_weight=w,
                categorical_feature=["shop_id","type_id","fut_w","fut_m"]
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            m = GradientBoostingRegressor(
                loss="squared_error", n_estimators=300, learning_rate=0.05,
                max_depth=3, random_state=SEED
            )
            m.fit(X, y, sample_weight=w.values if hasattr(w,"values") else w)
        models[h] = m
    return models



def predict_group_hybrid(model, gbm_models, scaler, shop_enc, type_enc, g):
    lstm_pred, feats = predict_group_lstm(model, scaler, shop_enc, type_enc, g)
    log_lstm = np.log1p(lstm_pred)

    res = []
    for t in range(PREDICT):
        row = {
            "shop_id": feats["shop_id"], "type_id": feats["type_id"],
            "fut_w": int(feats["w_f"][t]), "fut_m": int(feats["m_f"][t]),
            "last_s": feats["last_s"], "mean7_s": feats["mean7_s"],
            "mean28_s": feats["mean28_s"], "trend_s": feats["trend_s"],
            "log_lstm": float(log_lstm[t]),
        }
        X = pd.DataFrame([row], columns=GBM_FEATURES)
        # ★ 학습 때와 ‘동일 카테고리 집합’으로 캐스팅
        X = cast_gbm_dtypes(X, shop_enc, type_enc)

        # 저수요는 보정 skip (선택)
        if lstm_pred[t] < 1.0:
            res.append(0.0); continue

        delta = float(gbm_models[t+1].predict(X)[0])
        # 필요시 클립/베타 적용
        delta = np.clip(delta, -DELTA_CLIP, DELTA_CLIP) * RESID_BETA
        res.append(delta)

    log_hat = log_lstm + np.array(res, dtype=np.float32)
    final = np.expm1(log_hat)
    return np.clip(final, 0, None)

"""# Inference helpers

"""

def make_future_calendar(last_date, horizon=7):
    days = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    w = days.weekday.values.astype(int)
    m = (days.month.values.astype(int) - 1)
    return w, m

@torch.no_grad()
def predict_group_lstm(model, scaler, shop_enc, type_enc, g):
    g = g.sort_values("영업일자").copy()
    assert len(g) >= LOOKBACK
    g_look = g.tail(LOOKBACK).copy()

    w_p = g_look["요일"].values.astype(int)
    m_p = g_look["개월"].values.astype(int)
    last_date = g_look["영업일자"].max()
    w_f, m_f = make_future_calendar(last_date, PREDICT)

    x_vals = g_look["매출수량"].values.astype(float)
    x_scaled = scaler.transform(x_vals).astype(np.float32)

    shop, menu = split_shop_menu(g_look["영업장명_메뉴명"].iloc[0])
    sid = shop_enc.map.get(shop, shop_enc.unk_index())
    mtype = categorize_menu(menu)
    tid = type_enc.map.get(mtype, type_enc.unk_index())

    stats = np.array([x_scaled[-1], x_scaled[-7:].mean(), x_scaled.mean()], np.float32)
    x_t   = torch.tensor(x_scaled).unsqueeze(0).unsqueeze(-1).to(DEVICE)
    w_p_t = torch.tensor(w_p, dtype=torch.long).unsqueeze(0).to(DEVICE)
    m_p_t = torch.tensor(m_p, dtype=torch.long).unsqueeze(0).to(DEVICE)
    w_f_t = torch.tensor(w_f, dtype=torch.long).unsqueeze(0).to(DEVICE)
    m_f_t = torch.tensor(m_f, dtype=torch.long).unsqueeze(0).to(DEVICE)
    sid_t = torch.tensor([sid], dtype=torch.long).to(DEVICE)
    tid_t = torch.tensor([tid], dtype=torch.long).to(DEVICE)
    st_t  = torch.tensor(stats).unsqueeze(0).to(DEVICE)

    model.eval()
    pred_scaled = model(x_t, w_p_t, m_p_t, w_f_t, m_f_t, sid_t, tid_t, st_t).cpu().numpy().reshape(-1)
    pred = np.expm1(pred_scaled)  # 원단위
    pred = np.clip(pred, 0, None)
    # plus features for LGBM
    base_feats = {
        "shop_id": sid, "type_id": tid,
        "last_s": float(stats[0]), "mean7_s": float(stats[1]), "mean28_s": float(stats[2]),
        "trend_s": float(stats[0]-stats[1]),
        "ratio_l_m7": float(stats[0]/(abs(stats[1])+1e-6)),
        "w_f": w_f, "m_f": m_f
    }
    return pred, base_feats

# def main():
#     # LSTM 학습 (Stage A)
#     model, scaler, shop_enc, type_enc = train_and_build_lstm(TRAIN_PATH)

#     # LGBM 잔차 학습 (Stage B) — train.csv만 사용
#     tr_for_gbm = add_features(handle_negatives_train(pd.read_csv(TRAIN_PATH), NEGATIVE_STRATEGY_TRAIN))
#     gbm_models = train_lgbm_residual(tr_for_gbm, model, scaler, shop_enc, type_enc)

#     # 추론
#     sample = pd.read_csv(SAMPLE_SUB)
#     all_rows = []
#     test_files = sorted(glob.glob(f"{TEST_DIR}/TEST_*.csv"))

#     for tp in test_files:
#         test_id = os.path.basename(tp).split(".")[0]
#         df = pd.read_csv(tp)
#         # 캘린더만 추가 (입력 28일 그대로 사용)
#         df["영업일자"] = pd.to_datetime(df["영업일자"], errors="coerce")
#         df["요일"] = df["영업일자"].dt.weekday.astype(int).clip(0,6)
#         df["개월"] = (df["영업일자"].dt.month-1).astype(int).clip(0,11)

#         for key, g in df.groupby("영업장명_메뉴명"):
#             if len(g) < LOOKBACK:
#                 continue
#             pred7 = predict_group_hybrid(model, gbm_models, scaler, shop_enc, type_enc, g)
#             for d, v in enumerate(pred7, start=1):
#                 all_rows.append({
#                     "영업일자": f"{test_id}+{d}일",
#                     "영업장명_메뉴명": key,
#                     "매출수량": float(v)
#                 })

#     # 제출 파일 생성
#     pred_long = pd.DataFrame(all_rows)
#     pred_wide = pred_long.pivot(index="영업일자",
#                                 columns="영업장명_메뉴명",
#                                 values="매출수량").reset_index()
#     for col in sample.columns:
#         if col not in pred_wide.columns:
#             pred_wide[col] = 0
#     pred_wide = pred_wide[sample.columns]
#     pred_wide.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
#     print("Saved:", OUT_PATH)

# if __name__ == "__main__":
#     main()

def smape_np(y_true, y_pred, eps=1e-6):
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    mask = (np.abs(y_true) > 0).astype(float)
    term = 2.0 * np.abs(y_pred - y_true) / denom
    per_sample = (term * mask).sum() / mask.sum().clip(min=1.0)
    return per_sample

def main():
    # LSTM 학습 (Stage A)
    model, scaler, shop_enc, type_enc = train_and_build_lstm(TRAIN_PATH)

    # LGBM 잔차 학습 (Stage B) — train.csv만 사용
    tr_for_gbm = add_features(handle_negatives_train(pd.read_csv(TRAIN_PATH), NEGATIVE_STRATEGY_TRAIN))
    gbm_models = train_lgbm_residual(tr_for_gbm, model, scaler, shop_enc, type_enc)

    # 추론
    sample = pd.read_csv(SAMPLE_SUB)
    all_rows = []
    test_files = sorted(glob.glob(f"{TEST_DIR}/TEST_*.csv"))

    for tp in test_files:
        test_id = os.path.basename(tp).split(".")[0]
        df = pd.read_csv(tp)
        # 캘린더만 추가 (입력 28일 그대로 사용)
        df["영업일자"] = pd.to_datetime(df["영업일자"], errors="coerce")
        df["요일"] = df["영업일자"].dt.weekday.astype(int).clip(0,6)
        df["개월"] = (df["영업일자"].dt.month-1).astype(int).clip(0,11)

        file_true = []
        file_pred = []

        for key, g in df.groupby("영업장명_메뉴명"):
            if len(g) < LOOKBACK:
                continue
            pred7 = predict_group_hybrid(model, gbm_models, scaler, shop_enc, type_enc, g)
            true7 = g.tail(PREDICT)["매출수량"].values.astype(float)
            if len(true7) == PREDICT:
                file_true.extend(true7)
                file_pred.extend(pred7)
            for d, v in enumerate(pred7, start=1):
                all_rows.append({
                    "영업일자": f"{test_id}+{d}일",
                    "영업장명_메뉴명": key,
                    "매출수량": float(v)
                })
        # SMAPE 계산 및 출력
        if len(file_true) > 0 and len(file_pred) > 0:
            smape_val = smape_np(np.array(file_true), np.array(file_pred))
            print(f"[{tp}] SMAPE: {smape_val:.4f}")

    # 제출 파일 생성
    pred_long = pd.DataFrame(all_rows)
    pred_wide = pred_long.pivot(index="영업일자",
                                columns="영업장명_메뉴명",
                                values="매출수량").reset_index()
    for col in sample.columns:
        if col not in pred_wide.columns:
            pred_wide[col] = 0
    pred_wide = pred_wide[sample.columns]
    pred_wide.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()