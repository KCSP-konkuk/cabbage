"""
배추 도매 평균가격 순별 예측 모델 (최종 버전)
==============================================
모델: XGBoost
피쳐: 전체 77개 중 importance 기반 동적 선별
학습: 2018~2024, 테스트: 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import time

t0 = time.time()
DATA_DIR = "./"  # 데이터 파일이 있는 경로


# ================================================================
# 1. 데이터 로드
# ================================================================
def parse_date(d):
    """'202301상순' → (2023, 1, 0, '상순')"""
    d = d.strip().replace('\ufeff', '')
    period_map = {'상순': 0, '중순': 1, '하순': 2}
    year = int(d[:4])
    month = int(d[4:6])
    period_str = d[6:]
    period = period_map[period_str]
    return year, month, period, period_str


def to_idx(year, month, period):
    return (year - 2017) * 36 + (month - 1) * 3 + period


def load_csv(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    parsed = df['DATE'].apply(parse_date)
    df['Year'] = parsed.apply(lambda x: x[0])
    df['Month'] = parsed.apply(lambda x: x[1])
    df['Period'] = parsed.apply(lambda x: x[2])
    df['PeriodStr'] = parsed.apply(lambda x: x[3])
    df['idx'] = df.apply(lambda r: to_idx(r['Year'], r['Month'], r['Period']), axis=1)
    return df.sort_values('idx').reset_index(drop=True)


print("=" * 60)
print("  배추 도매가격 순별 예측 모델")
print("=" * 60)

price = load_csv(DATA_DIR + "가격데이터_순별.csv")
for col in ['평균가격', '전년', '평년']:
    price[col] = pd.to_numeric(price[col], errors='coerce')

haenam = load_csv(DATA_DIR + "Haenam_순별.csv")
taebaek = load_csv(DATA_DIR + "TaeBak_순별.csv")

supply = load_csv(DATA_DIR + "배추반입량_순별.csv")
supply['총반입량'] = pd.to_numeric(supply['총반입량'], errors='coerce')

search = load_csv(DATA_DIR + "search_순별.csv")
search['평균_검색량'] = pd.to_numeric(search['평균_검색량'], errors='coerce')

print(f"[로드] 가격 {price.shape[0]}행 | 해남 {haenam.shape[0]}행 | "
      f"태백 {taebaek.shape[0]}행 | 반입량 {supply.shape[0]}행 | 검색량 {search.shape[0]}행")


# ================================================================
# 2. 날씨 병합 (7~9월 태백, 나머지 해남)
# ================================================================
WEATHER_COLS = [
    '평균_최저기온', '평균_최고기온', '총_강수량', '평균_풍속',
    '평균_상대습도', '평균_일사량', '평균_지면온도', '평균_최저초상온도'
]

h_dict = haenam.set_index('idx')
t_dict = taebaek.set_index('idx')

weather_rows = []
for idx in sorted(set(h_dict.index) | set(t_dict.index)):
    h = h_dict.loc[idx] if idx in h_dict.index else None
    t = t_dict.loc[idx] if idx in t_dict.index else None
    month = h['Month'] if h is not None else t['Month']

    if month in [7, 8, 9] and t is not None:
        src = t
    else:
        src = h if h is not None else t

    row = {'idx': idx}
    for c in WEATHER_COLS:
        row[c] = src[c] if src is not None else np.nan
    weather_rows.append(row)

weather = pd.DataFrame(weather_rows)


# ================================================================
# 3. 전체 병합
# ================================================================
df = price[['idx', 'Year', 'Month', 'Period', 'PeriodStr',
            '평균가격', '전년', '평년', 'DATE']].copy()
df = df.merge(weather[['idx'] + WEATHER_COLS], on='idx', how='left')
df = df.merge(supply[['idx', '총반입량']], on='idx', how='left')
df = df.merge(search[['idx', '평균_검색량']], on='idx', how='left')
df = df.sort_values('idx').reset_index(drop=True)
print(f"[병합] {df.shape[0]}행 × {df.shape[1]}열")


# ================================================================
# 4. 피쳐 엔지니어링 (77개 후보 생성)
# ================================================================
TARGET = '평균가격'

# 가격 lag
for lag in [1, 2, 3, 4, 5, 6, 9, 12, 18, 36]:
    df[f'plag{lag}'] = df[TARGET].shift(lag)

# 가격 이동평균
for w in [3, 6, 12]:
    df[f'pma{w}'] = df[TARGET].shift(1).rolling(w).mean()

# 가격 변동성
for w in [3, 6]:
    df[f'pstd{w}'] = df[TARGET].shift(1).rolling(w).std()

# 모멘텀, YoY
df['pmom3'] = df['plag1'] - df['plag4']
df['pyoy'] = df[TARGET].shift(1) / df[TARGET].shift(37) - 1

# 가격 vs 평년/전년
df['p_vs_py'] = df['plag1'] / df['평년'].replace(0, np.nan)
df['p_vs_jn'] = df['plag1'] / df['전년'].replace(0, np.nan)

# 반입량
for lag in [1, 2, 3]:
    df[f'slag{lag}'] = df['총반입량'].shift(lag)
df['sma3'] = df['총반입량'].shift(1).rolling(3).mean()
df['sma6'] = df['총반입량'].shift(1).rolling(6).mean()
df['schg'] = df['총반입량'].shift(1).pct_change()
df['svma'] = df['slag1'] / df['sma6'].replace(0, np.nan)
df['ps_ratio'] = df['plag1'] / df['slag1'].replace(0, np.nan)

# 검색량
for lag in [1, 3, 6, 12, 18]:
    df[f'srlag{lag}'] = df['평균_검색량'].shift(lag)
df['srma3'] = df['평균_검색량'].shift(1).rolling(3).mean()

# 날씨 lag
for c in WEATHER_COLS:
    for lag in [1, 3, 6, 9]:
        df[f'{c}_l{lag}'] = df[c].shift(lag)

# 날씨 파생
df['temp_range_l3'] = df['평균_최고기온'].shift(3) - df['평균_최저기온'].shift(3)
df['heat_stress'] = (df['평균_최고기온'].shift(3) > 30).astype(int)
df['cold_stress'] = (df['평균_최저기온'].shift(3) < -5).astype(int)
df['heavy_rain_l3'] = (df['총_강수량'].shift(3) > 100).astype(int)
df['heavy_rain_l6'] = (df['총_강수량'].shift(6) > 100).astype(int)

# 달력 (sin/cos)
df['msin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['mcos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['piy'] = (df['Month'] - 1) * 3 + df['Period']
df['pysin'] = np.sin(2 * np.pi * df['piy'] / 36)
df['pycos'] = np.cos(2 * np.pi * df['piy'] / 36)
df['kimchi'] = ((df['Month'] >= 10) & (df['Month'] <= 12)).astype(int)
df['summer'] = ((df['Month'] >= 7) & (df['Month'] <= 9)).astype(int)


# ================================================================
# 5. 데이터 정제 + Train/Test 분리
# ================================================================
EXCLUDE = (
    {'idx', 'Year', 'Month', 'Period', 'PeriodStr', 'DATE',
     TARGET, '전년', '평년', '총반입량', '평균_검색량'}
    | set(WEATHER_COLS)
)
ALL_FEATURES = [c for c in df.columns
                if c not in EXCLUDE and not df[c].isna().all()]

first_valid = df[ALL_FEATURES].dropna().index.min()
df_clean = df.loc[first_valid:].copy()
df_clean[ALL_FEATURES] = (
    df_clean[ALL_FEATURES]
    .fillna(method='ffill').fillna(0)
    .replace([np.inf, -np.inf], 0)
)

train = df_clean[(df_clean['Year'] >= 2018) & (df_clean['Year'] <= 2024)]
test = df_clean[df_clean['Year'] == 2025]

print(f"[후보 피쳐] {len(ALL_FEATURES)}개")
print(f"[학습] {len(train)}행 (2018~2024)")
print(f"[테스트] {len(test)}행 (2025)")


# ================================================================
# 6. 피쳐 선별 (importance 기반 32개)
# ================================================================
print("\n--- 피쳐 선별 (importance 기반) ---")

y_tr = train[TARGET].values
y_te = test[TARGET].values

# importance 기반 선별 32개 피쳐
SELECTED = [
    'pma3', 'pmom3', 'plag1', '평균_최저초상온도_l3', 'piy', 'pycos',
    '평균_최저기온_l3', 'pysin', 'msin', '평균_최고기온_l3', 'summer',
    '평균_지면온도_l3', '평균_최저기온_l9', '평균_지면온도_l6', 'pma6', 'plag4',
    'sma3', 'plag9', 'plag2', 'heavy_rain_l3', 'plag3', 'pyoy',
    'srlag12', '평균_최고기온_l1', 'pstd6', 'ps_ratio', 'heavy_rain_l6',
    '평균_일사량_l9', '평균_최저초상온도_l9', '총_강수량_l3', 'plag36', '평균_최고기온_l6',
]
best_n = len(SELECTED)
print(f"  선별 피쳐 수: {best_n}개")

X_train = train[SELECTED].values
X_test = test[SELECTED].values


# ================================================================
# 7. 모델 학습
# ================================================================
print("\n" + "=" * 60)
print("  모델 학습: XGBoost")
print("=" * 60)

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)
model.fit(X_train, y_tr)
print("  학습 완료!")


# ================================================================
# 8. 교차검증 (TimeSeriesSplit 5-fold)
# ================================================================
print("\n--- TimeSeriesSplit 교차검증 (5-fold) ---")
tscv = TimeSeriesSplit(n_splits=5)
cv_results = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    fold_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    fold_model.fit(X_train[tr_idx], y_tr[tr_idx])
    fold_pred = fold_model.predict(X_train[val_idx])

    fold_r2 = r2_score(y_tr[val_idx], fold_pred)
    fold_rmse = np.sqrt(mean_squared_error(y_tr[val_idx], fold_pred))
    cv_results.append({'fold': fold, 'R2': fold_r2, 'RMSE': fold_rmse})
    print(f"  Fold {fold}: R²={fold_r2:.4f}, RMSE={fold_rmse:,.0f}")

avg_r2 = np.mean([r['R2'] for r in cv_results])
avg_rmse = np.mean([r['RMSE'] for r in cv_results])
print(f"  ─────────────────────────────")
print(f"  평균:  R²={avg_r2:.4f}, RMSE={avg_rmse:,.0f}")


# ================================================================
# 9. 테스트 평가 (2025)
# ================================================================
y_pred = model.predict(X_test)

test_r2 = r2_score(y_te, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
test_mae = mean_absolute_error(y_te, y_pred)
test_mape = np.mean(np.abs((y_te - y_pred) / y_te)) * 100

print("\n" + "=" * 60)
print("  2025년 테스트 결과")
print("=" * 60)
print(f"  R²:   {test_r2:.4f}")
print(f"  RMSE: {test_rmse:,.0f}")
print(f"  MAE:  {test_mae:,.0f}")
print(f"  MAPE: {test_mape:.1f}%")
print("=" * 60)


# ================================================================
# 10. 순별 예측 상세
# ================================================================
print(f"\n{'DATE':>14s}  {'실제':>10s}  {'예측':>10s}  {'오차':>10s}  {'오차율':>7s}")
print("-" * 58)

for i in range(len(test)):
    date_str = test.iloc[i]['DATE']
    actual = y_te[i]
    pred = y_pred[i]
    error = actual - pred
    pct = abs(error) / actual * 100
    print(f"{date_str:>14s}  {actual:>10,.0f}  {pred:>10,.0f}  {error:>+10,.0f}  {pct:>6.1f}%")


# ================================================================
# 11. 피쳐 중요도 (Top 15)
# ================================================================
final_imp = pd.DataFrame({
    'feature': SELECTED,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False)

print(f"\n--- 피쳐 중요도 (Top 15 / 전체 {best_n}개) ---")
for _, row in final_imp.head(15).iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:25s}  {row['importance']:.4f}  {bar}")


# ================================================================
# 12. 시각화 (6개 플롯)
# ================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import platform

    # 한글 폰트 설정
    if platform.system() == 'Windows':
        # Windows: 맑은 고딕 우선, 없으면 나눔고딕
        font_candidates = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Gulim']
        font_set = False
        for font_name in font_candidates:
            font_list = [f.name for f in fm.fontManager.ttflist]
            if font_name in font_list:
                plt.rcParams['font.family'] = font_name
                font_set = True
                print(f"  [폰트] {font_name} 사용")
                break
        if not font_set:
            # 시스템 폰트 직접 경로 탐색
            import os
            win_font_dir = 'C:/Windows/Fonts'
            for fname in ['malgun.ttf', 'malgunbd.ttf', 'NanumGothic.ttf', 'gulim.ttc']:
                fpath = os.path.join(win_font_dir, fname)
                if os.path.exists(fpath):
                    fm.fontManager.addfont(fpath)
                    prop = fm.FontProperties(fname=fpath)
                    plt.rcParams['font.family'] = prop.get_name()
                    print(f"  [폰트] {fpath} 직접 로드")
                    font_set = True
                    break
            if not font_set:
                print("  [폰트] 한글 폰트를 찾지 못했습니다. 깨질 수 있습니다.")
    elif platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        try:
            import koreanize_matplotlib
        except ImportError:
            plt.rcParams['font.family'] = 'NanumGothic'

    plt.rcParams['axes.unicode_minus'] = False

    COLORS = {
        'actual': '#2C3E50', 'pred': '#FF5722', 'fill': '#FF5722',
        'train': '#3498DB', 'bar': '#FF7043', 'res_pos': '#E74C3C',
        'res_neg': '#2ECC71', 'cv_bar': '#42A5F5', 'cv_avg': '#FF5722',
    }

    # ── ① 2025 실제 vs 예측 ──
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(test))
    ax.plot(x, y_te, 'o-', color=COLORS['actual'], linewidth=2.2,
            markersize=7, label='실제가격', zorder=5)
    ax.plot(x, y_pred, 's-', color=COLORS['pred'], linewidth=2,
            markersize=5, label='XGBoost 예측', zorder=4)
    ax.fill_between(x, y_te, y_pred, alpha=0.12, color=COLORS['fill'])

    for i in range(len(test)):
        err_pct = abs(y_te[i] - y_pred[i]) / y_te[i] * 100
        ax.annotate(f'{err_pct:.1f}%', (i, (y_te[i] + y_pred[i]) / 2),
                    fontsize=7, ha='center', color='gray', alpha=0.8)

    xlabels = [d.replace('2025', "'25") for d in test['DATE']]
    ax.set_xticks(range(len(test)))
    ax.set_xticklabels(xlabels, rotation=55, ha='right', fontsize=8)
    ax.set_title(
        f'2025 배추 도매가격: 실제 vs 예측  |  '
        f'R²={test_r2:.4f}  RMSE={test_rmse:,.0f}  MAPE={test_mape:.1f}%',
        fontsize=13, fontweight='bold', pad=12,
    )
    ax.set_ylabel('가격 (원/10kg)', fontsize=11)
    ax.set_xlabel('순별', fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(DATA_DIR + 'result_2025_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [1/6] result_2025_prediction.png")

    # ── ② 산점도 ──
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_te, y_pred, s=70, c=COLORS['pred'], alpha=0.75,
               edgecolors='white', linewidth=0.8, zorder=5)
    mn = min(y_te.min(), y_pred.min()) * 0.9
    mx = max(y_te.max(), y_pred.max()) * 1.1
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.4, linewidth=1, label='y = x (완벽 예측)')
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_xlabel('실제 가격 (원)', fontsize=11)
    ax.set_ylabel('예측 가격 (원)', fontsize=11)
    ax.set_title(f'실제 vs 예측 산점도 (R²={test_r2:.4f})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(DATA_DIR + 'result_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [2/6] result_scatter.png")

    # ── ③ 피쳐 중요도 (Top 20) ──
    top_n = min(20, len(final_imp))
    top_imp = final_imp.head(top_n).iloc[::-1]  # 역순 (아래→위)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(top_n), top_imp['importance'].values,
                   color=COLORS['bar'], alpha=0.85, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_imp['feature'].values, fontsize=9)
    for i, v in enumerate(top_imp['importance'].values):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=8, color='gray')
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(f'피쳐 중요도 Top {top_n} (전체 {best_n}개 선택)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(DATA_DIR + 'result_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [3/6] result_feature_importance.png")

    # ── ④ 전체 시계열 (2018~2025) ──
    full_pred = model.predict(df_clean[SELECTED].values)

    fig, ax = plt.subplots(figsize=(16, 6))
    dates = df_clean['DATE'].values
    actual_vals = df_clean[TARGET].values
    n_pts = len(dates)
    x_full = range(n_pts)

    # 학습/테스트 경계
    n_train = len(train)
    ax.axvline(x=n_train - 0.5, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
    ax.axvspan(n_train - 0.5, n_pts, alpha=0.06, color='orange')
    ax.text(n_train + 1, max(actual_vals) * 0.95, '← 2025 테스트',
            fontsize=9, color='gray', style='italic')

    ax.plot(x_full, actual_vals, '-', color=COLORS['actual'], linewidth=1.5,
            alpha=0.8, label='실제가격')
    ax.plot(x_full, full_pred, '-', color=COLORS['pred'], linewidth=1.2,
            alpha=0.65, label='모델 예측')

    # x축 레이블 (연도 시작만)
    tick_pos, tick_lab = [], []
    for i, d in enumerate(dates):
        if '01상순' in str(d):
            tick_pos.append(i)
            tick_lab.append(str(d)[:4])
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=10)

    ax.set_title('배추 도매가격 전체 시계열 (2018~2025)', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('가격 (원/10kg)', fontsize=11)
    ax.set_xlabel('연도', fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(DATA_DIR + 'result_full_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [4/6] result_full_timeseries.png")

    # ── ⑤ 잔차 분석 ──
    residuals = y_te - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 좌: 잔차 바 차트
    ax1 = axes[0]
    colors_r = [COLORS['res_neg'] if r < 0 else COLORS['res_pos'] for r in residuals]
    ax1.bar(range(len(residuals)), residuals, color=colors_r, alpha=0.75, edgecolor='white')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_xticks(range(len(test)))
    ax1.set_xticklabels(xlabels, rotation=55, ha='right', fontsize=7)
    ax1.set_ylabel('잔차 (실제 - 예측)', fontsize=10)
    ax1.set_title('순별 예측 잔차', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 우: 잔차 히스토그램
    ax2 = axes[1]
    ax2.hist(residuals, bins=min(12, len(residuals)), color=COLORS['pred'],
             alpha=0.7, edgecolor='white')
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.axvline(np.mean(residuals), color='blue', linewidth=1.2, linestyle='-',
                label=f'평균={np.mean(residuals):,.0f}')
    ax2.set_xlabel('잔차 (원)', fontsize=10)
    ax2.set_ylabel('빈도', fontsize=10)
    ax2.set_title('잔차 분포', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--')

    plt.suptitle(f'잔차 분석  |  MAE={test_mae:,.0f}원  MAPE={test_mape:.1f}%',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(DATA_DIR + 'result_residuals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [5/6] result_residuals.png")

    # ── ⑥ CV 결과 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = [r['fold'] for r in cv_results]
    r2s = [r['R2'] for r in cv_results]

    bar_colors = [COLORS['cv_bar'] if r >= 0 else '#EF9A9A' for r in r2s]
    ax.bar(folds, r2s, color=bar_colors, alpha=0.8, edgecolor='white', width=0.6)
    ax.axhline(avg_r2, color=COLORS['cv_avg'], linewidth=1.8, linestyle='--',
               label=f'CV 평균 R²={avg_r2:.4f}')
    ax.axhline(test_r2, color='green', linewidth=1.5, linestyle=':',
               label=f'테스트 R²={test_r2:.4f}')

    for i, r in enumerate(r2s):
        ax.text(folds[i], r + 0.02 if r >= 0 else r - 0.06,
                f'{r:.3f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('TimeSeriesSplit 5-Fold 교차검증 결과', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticks(folds)
    plt.tight_layout()
    plt.savefig(DATA_DIR + 'result_cv_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [6/6] result_cv_results.png")

    print(f"\n[시각화 완료] result_*.png 6개 파일 저장됨")
except Exception as e:
    print(f"\n[시각화 건너뜀] {e}")
    import traceback; traceback.print_exc()


# ================================================================
# 13. 결과 저장
# ================================================================
result_df = test[['DATE', 'Year', 'Month', 'Period', TARGET]].copy()
result_df['예측가격'] = y_pred
result_df['오차'] = y_te - y_pred
result_df['오차율(%)'] = np.abs(result_df['오차'] / y_te) * 100
result_df.to_csv(DATA_DIR + 'result_2025_predictions.csv',
                 index=False, encoding='utf-8-sig')

print(f"[저장] result_2025_predictions.csv")
print(f"\n총 소요시간: {time.time() - t0:.1f}초")
