import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from google.colab import drive
drive.mount('/content/drive')

# 한글 폰트 설정
font_path = '/content/drive/MyDrive/SeoulNamsanC.otf'
font_prop = fm.FontProperties(fname=font_path)
plt.style.use('dark_background')

# --- 1. 데이터 로딩 및 전처리 ---
print("데이터 로딩 중...")
train_df = pd.read_csv('/content/drive/MyDrive/lgaimers/train.csv')

# 음수데이터 처리
train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
# '영업장명_메뉴명' 컬럼에 '단체'가 포함된 행만 남김
train_df = train_df[train_df['영업장명_메뉴명'].str.contains('단체')]

# 입력 및 예측 구간 데이터 로딩 + 이상치 필터링 (단체식 중심으로 학습 및 예측)
test_df = pd.read_csv('/content/drive/MyDrive/lgaimers/TEST_00.csv')
test_df = test_df[test_df['영업장명_메뉴명'].str.contains('단체')]

train_df['영업일자'] = pd.to_datetime(train_df['영업일자'])
test_df['영업일자'] = pd.to_datetime(test_df['영업일자'])

print("데이터 피벗 테이블 생성 중...")
sales_df = train_df.pivot_table(index='영업일자', columns='영업장명_메뉴명', values='매출수량', fill_value=0)
sales_df = sales_df.asfreq('D', fill_value=0)

time_features_df = pd.DataFrame(index=sales_df.index)
time_features_df['month'] = sales_df.index.month
time_features_df['dayofweek'] = sales_df.index.dayofweek

time_features_df['is_weekend'] = time_features_df['dayofweek'].isin([5, 6]).astype(int)
dow_onehot = pd.get_dummies(time_features_df['dayofweek'], prefix='dow')
time_features_df = pd.concat([time_features_df, dow_onehot], axis=1)
BASE_COLS = ['is_weekend','month'] + [f'dow_{d}' for d in range(7)]

time_features_df['is_weekend_dup'] = time_features_df['is_weekend']
BASE_COLS.append('is_weekend_dup')
print("데이터 전처리 완료.")


# --- 2. 모델 클래스 정의 ---
class BlockBootstrapRandomForest:
    """
    Random Forest Regressor with a custom blocked bootstrap sampling method.
    """
    def __init__(self, n_estimators=100, max_depth=10, block_size=28, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.block_size = block_size
        self.random_state = random_state
        self.trees = []
        np.random.seed(self.random_state)

    def _blocked_bootstrap_indices(self, n_samples):
        num_blocks = n_samples - self.block_size + 1
        if num_blocks <= 0:
            return np.random.choice(n_samples, size=n_samples, replace=True)
        block_starts = np.arange(num_blocks)
        random_block_starts = np.random.choice(block_starts, size=num_blocks, replace=True)
        bootstrap_indices = []
        for start in random_block_starts:
            bootstrap_indices.extend(np.arange(start, start + self.block_size))
        return np.array(bootstrap_indices[:n_samples])

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for i in tqdm(range(self.n_estimators), desc=f"  - {self.n_estimators}개 나무 학습"):
            boot_indices = self._blocked_bootstrap_indices(n_samples)
            if len(boot_indices) == 0: continue
            X_boot, y_boot = X[boot_indices], y[boot_indices]
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=i
                )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict(self, X):
        if not self.trees:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        X = X.reshape(1, -1)
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0).flatten()


# --- 3. 피처 엔지니어링 및 학습/예측 함수 ---

def create_sliding_window_with_stats_and_lags(
    sales_data: np.ndarray,
    time_features: pd.DataFrame,
    look_back: int,
    horizon: int
):
    """
    매출수량, 통계량, 래그 피처, 시간 특성을 결합하여 슬라이딩 윈도우 생성.
    """
    X, y_raw = [], []

    for i in range(len(sales_data) - look_back - horizon + 1):
        # 윈도우 구간의 매출 및 통계량
        window_sales = sales_data[i : i + look_back]
        mean_lag = np.mean(window_sales)
        std_lag = np.std(window_sales)

        # 래그 피처 추가 (i + look_back 는 현재 시점의 인덱스)
        lag_1 = sales_data[i + look_back - 1] if i + look_back - 1 >= 0 else 0
        lag_7 = sales_data[i + look_back - 7] if i + look_back - 7 >= 0 else 0
        lag_14 = sales_data[i + look_back - 14] if i + look_back - 14 >= 0 else 0

        # 예측 구간의 미래 날짜 특성
        future_time = (
            time_features.iloc[i + look_back : i + look_back + horizon][BASE_COLS]
            .values
            .flatten()
        )

        # 모든 피처를 결합 (래그 피처 추가)
        feats = np.concatenate([window_sales, [mean_lag, std_lag, lag_1, lag_7, lag_14], future_time])
        X.append(feats)

        # 정답(y) -> 다음 horizon일 매출
        y_raw.append(sales_data[i + look_back : i + look_back + horizon])
    
    return np.asarray(X), np.asarray(y_raw)


def two_stage_train_and_predict(
    historical_series: pd.Series,
    time_features: pd.DataFrame,
    test_sales_input: np.ndarray,
    test_time_input: np.ndarray
):
    LOOK_BACK = 28
    HORIZON = 7
    BLOCK_SIZE = 28

    print(f"\n--- '{historical_series.name}' 2단계 예측 모델 시작 ---")

    # 1. 학습 데이터 생성 (래그 피처 포함)
    X_train_raw, y_train_raw = create_sliding_window_with_stats_and_lags(
        historical_series.values, time_features, LOOK_BACK, HORIZON
    )
    if len(X_train_raw) < BLOCK_SIZE:
        print("  - 데이터 부족. 0으로 예측합니다.")
        return np.zeros(HORIZON, dtype=int)

    # 1단계: 매출 발생 여부 (분류) 모델 학습
    print("  - 1단계: 매출 발생 분류 모델 학습...")
    y_train_cls = np.clip(y_train_raw, 0, 1).astype(int)
    
    # y_train_cls는 7일치 예측이므로, 7개의 트리 모델을 학습
    cls_models = []
    for i in range(HORIZON):
        model = DecisionTreeClassifier(max_depth=5, random_state=i)
        model.fit(X_train_raw, y_train_cls[:, i])
        cls_models.append(model)
        
    # 2단계: 매출 수량 (회귀) 모델 학습
    print("  - 2단계: 매출 수량 회귀 모델 학습...")
    # 매출이 0이 아닌 데이터만 필터링
    non_zero_indices = np.where(np.any(y_train_raw > 0, axis=1))[0]
    X_train_reg = X_train_raw[non_zero_indices]
    y_train_reg = y_train_raw[non_zero_indices]

    reg_model = BlockBootstrapRandomForest(
        ### 랜덤포레스트 회귀모델 파라미터 설정
        n_estimators=100, max_depth=10, block_size=BLOCK_SIZE, random_state=42
    )
    if len(X_train_reg) < BLOCK_SIZE:
        print("  - 2단계 회귀 모델 학습을 위한 데이터 부족. 1단계 예측만 사용합니다.")
        reg_model_is_trained = False
    else:
        reg_model.fit(X_train_reg, y_train_reg)
        reg_model_is_trained = True

    # 3. 예측 과정
    print("  - 미래 7일 예측 시작...")
    
    # 테스트 입력 데이터 준비 (래그 피처 포함)
    mean_test = np.mean(test_sales_input)
    std_test  = np.std(test_sales_input)
    
    lag_1_test = test_sales_input[-1]
    lag_7_test = test_sales_input[-7]
    lag_14_test = test_sales_input[-14]

    combined_test_input = np.concatenate([test_sales_input, [mean_test, std_test, lag_1_test, lag_7_test, lag_14_test], test_time_input.flatten()])

    final_predictions = np.zeros(HORIZON, dtype=int)
    
    for i in range(HORIZON):
        # 1단계: 매출 발생 여부 예측
        cls_pred = cls_models[i].predict(combined_test_input.reshape(1, -1))[0]

        if cls_pred == 1 and reg_model_is_trained:
            # 2단계: 매출이 발생한다고 예측되면, 수량 예측
            reg_pred = reg_model.predict(combined_test_input)
            final_predictions[i] = reg_pred[i]
    
    final_predictions = np.clip(np.round(final_predictions), 0, None).astype(int)

    print(f"--- '{historical_series.name}' 예측 완료 ---")
    return final_predictions


# --- 4. ★★★★★ 전체 프로세스 실행 ★★★★★

all_results = {}
test_df_grouped = test_df.groupby('영업장명_메뉴명')

# test_df_grouped이 (a,b) 형태의 튜플을 계속 반환할 때 각각 두 변수에 자동으로 할당한다. 반환되는 튜플의 형태는 여기서 (영업장명_메뉴명, 해당 품목의 데이터프레임)
for item, group in test_df_grouped:
    if item in sales_df.columns:
        # 1. 훈련 데이터: train.csv에서 해당 아이템의 전체 시계열
        full_historical_series = sales_df[item]

        # 2. ★★★★★ 데이터 분할 ★★★★★
        # 훈련 데이터에서 마지막 7일 분리
        adjusted_historical_series = full_historical_series[:-7]
        last_7_days_of_train = full_historical_series[-7:].values

        # 테스트 데이터에서 21일(입력용)과 7일(검증용) 분리
        sorted_group = group.sort_values('영업일자')
        first_21_days_of_test = sorted_group['매출수량'].iloc[:21].values
        actual_values = sorted_group['매출수량'].iloc[21:].values # 실제값 (정답)

        if len(first_21_days_of_test) != 21 or len(actual_values) != 7:
            print(f"경고: '{item}'의 테스트 데이터 분할 불가. 건너뜁니다.")
            continue

        # 3. ★★★★★ 검증용 입력 데이터 생성 ★★★★★
        validation_input_sales = np.concatenate([last_7_days_of_train, first_21_days_of_test])

        # 4. 예측에 필요한 미래 7일(검증 기간)의 시간 특성 생성
        validation_dates = sorted_group['영업일자'].iloc[21:]
        future_time_features = pd.DataFrame(index=validation_dates.index)
        future_time_features['month'] = validation_dates.dt.month
        future_time_features['dayofweek'] = validation_dates.dt.dayofweek
        future_time_features['is_weekend'] = future_time_features['dayofweek'].isin([5, 6]).astype(int)
        future_time_features['is_weekend_dup'] = future_time_features['is_weekend']
        ### 요일변수를 원핫인코딩으로 매핑
        dow_onehot_future = pd.get_dummies(future_time_features['dayofweek'], prefix='dow')
        future_time_features = pd.concat([future_time_features, dow_onehot_future], axis=1)

        validation_time_input = future_time_features[BASE_COLS].values

        # 5. 모델 훈련 및 예측 실행
        prediction = two_stage_train_and_predict(
            adjusted_historical_series,
            time_features_df,
            validation_input_sales,
            validation_time_input
        )

        # 결과 저장
        all_results[item] = {'prediction': prediction, 'actual': actual_values}

    else:
        print(f"경고: '{item}' 항목은 훈련 데이터에 없어 학습이 불가능합니다.")

# ★★★★★ 성능 평가 및 결과 시각화 ★★★★★

all_predictions_flat = []
all_actuals_flat = []

for item, result in all_results.items():
    all_predictions_flat.extend(result['prediction'])
    all_actuals_flat.extend(result['actual'])

# 전체 평균 절대 오차(MAE) 계산
if all_actuals_flat:
    total_mae = mean_absolute_error(all_actuals_flat, all_predictions_flat)
    print("\n\n" + "="*50)
    print(f"📊 전체 품목에 대한 최종 예측 성능 (MAE): {total_mae:.4f}")
    print("="*50 + "\n")