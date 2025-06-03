# ───── 0. SETUP ─────
import ast, warnings, json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# 한글 폰트 설정
from matplotlib import font_manager as fm 
for font in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    if any(font in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = font
        break

warnings.filterwarnings("ignore")
DATA = Path("./")
# 각 타겟별 별도의 출력 폴더 생성
OUT_PM25 = Path("./domestic_pm25")
OUT_PM10 = Path("./domestic_pm10")
OUT_PM25.mkdir(exist_ok=True)
OUT_PM10.mkdir(exist_ok=True)

# ───── 1. LOAD ─────
pm_kr_raw = pd.read_csv(DATA/"PM_KR_2015_2022.csv")
pm_cn_raw = pd.read_csv(DATA/"PM_CN_2015_2022.csv")
met_kr = pd.read_csv(DATA/"KOR_met_2015_2022.csv", parse_dates=["date"])
master = pd.read_csv(DATA/"df_master.csv")

# ───── 1-1. 공통 파싱 함수 ─────
def tidy_pm(df: pd.DataFrame) -> pd.DataFrame:
    """딕셔너리 문자열로 된 PM 값을 long-format & 숫자형으로 변환"""
    melted = df.melt(id_vars="date", var_name="city", value_name="raw")
    melted["date"] = pd.to_datetime(melted["date"])

    def extract(s: str, key: str):
        if not isinstance(s, str):                 return np.nan
        try:
            val = ast.literal_eval(s.replace("'", '"')).get(key, "").strip()
            return pd.to_numeric(val, errors="coerce")
        except Exception:                          return np.nan

    melted["pm25"] = melted["raw"].map(lambda x: extract(x, "pm25"))
    melted["pm10"] = melted["raw"].map(lambda x: extract(x, "pm10"))
    return melted.drop(columns="raw")

pm_kr = tidy_pm(pm_kr_raw)     # city=한국 시·도
pm_cn = tidy_pm(pm_cn_raw)     # city=중국 도시

# ───── 2. 시계열 보간 (city 단위) ─────
def interpolate_city(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c, g in df.groupby("city"):
        g = g.sort_values("date").set_index("date")
        g[["pm25", "pm10"]] = g[["pm25", "pm10"]].interpolate(method="time")
        g["city"] = c
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)

pm_kr = interpolate_city(pm_kr)
pm_cn = interpolate_city(pm_cn)

# ───── 3. 중국 lag 특성 (0~3일) // 미구현 ─────
pm_cn_curr = (
    pm_cn.pivot_table(index="date", columns="city", values=["pm25", "pm10"])
)
pm_cn_curr.columns = [f"{city}_{var}" for var, city in pm_cn_curr.columns] 
pm_cn_curr = pm_cn_curr.reset_index()

# ───── 4. master(연도) → 일 단위 보간 ─────
# PM10과 PM2.5 데이터 분리
master_pm10 = master[master['pm_type'] == 'pm10'].copy()
master_pm25 = master[master['pm_type'] == 'pm2.5'].copy()

# 날짜 변환 (먼저 수행)
master_pm10["date"] = pd.to_datetime(master_pm10["year"].astype(str) + "-01-01")
master_pm25["date"] = pd.to_datetime(master_pm25["year"].astype(str) + "-01-01")

# 불필요한 열 제거 (year, 총합, 농도, pm_type) - 날짜 변환 후 수행
drop_cols = ["year", "총합", "농도", "pm_type"]
master_pm10 = master_pm10.drop(columns=drop_cols)
master_pm25 = master_pm25.drop(columns=drop_cols)

# 일별 데이터로 보간하는 함수
def interpolate_to_daily(df):
    agg = (df.groupby(["시도", "date"]).mean(numeric_only=True)
            .reset_index()
            .rename(columns={"시도": "city"}))
    
    daily_blocks = []
    for c, g in agg.groupby("city"):
        g = g.set_index("date")
        
        # 다음 연도 1월 1일까지 일자 생성
        full_idx = pd.date_range(g.index.min(),
                    g.index.max() + pd.offsets.YearEnd(0),
                    freq="D")
        g = g.reindex(full_idx)
        
        # 선형 보간
        g = g.interpolate(method="linear")
        
        g["city"] = c
        daily_blocks.append(g.reset_index().rename(columns={"index": "date"}))
    
    return pd.concat(daily_blocks, ignore_index=True)

# PM10과 PM2.5 각각에 대해 일별 데이터로 보간
master_daily_pm10 = interpolate_to_daily(master_pm10)
master_daily_pm25 = interpolate_to_daily(master_pm25)

# ───── 5. 병합 ─────
# PM10 데이터셋 병합
df_pm10 = (pm_kr
        .merge(met_kr, on=["city", "date"], how="left")
        .merge(master_daily_pm10, on=["city", "date"], how="left")
        .merge(pm_cn_curr, on="date", how="left")
        )

# PM2.5 데이터셋 병합
df_pm25 = (pm_kr
        .merge(met_kr, on=["city", "date"], how="left")
        .merge(master_daily_pm25, on=["city", "date"], how="left")
        .merge(pm_cn_curr, on="date", how="left")
        )

# ───── 6. 회귀분석 함수 정의 ─────
def run_regression_analysis(df, target, output_dir):
    """
    지정된 타겟에 대한 회귀분석을 수행하고 결과를 저장하는 함수
    
    Parameters:
    -----------
    df : DataFrame
        분석에 사용할 데이터프레임
    target : str
        타겟 변수명 ('pm10' 또는 'pm25')
    output_dir : Path
        결과를 저장할 디렉토리 경로
    """
    print(f"\n===== {target.upper()} 회귀분석 시작 =====")
    
    # 타겟에 따라 제외할 컬럼 설정
    exclude_cols = ["pm25", "pm10", "date"]
    
    # 범주형 변수 설정
    df["city"] = df["city"].astype("category")
    
    # 특성 선택 - 타겟에 따라 관련 특성만 선택
    # PM10 관련 특성만 선택 (PM10 타겟인 경우)
    if target == "pm10":
        # 중국 도시의 PM2.5 관련 컬럼 제외
        pm25_cols = [col for col in df.columns if '_pm25' in col]
        exclude_cols.extend(pm25_cols)
    
    # PM2.5 관련 특성만 선택 (PM2.5 타겟인 경우)
    elif target == "pm25":
        # 중국 도시의 PM10 관련 컬럼 제외
        pm10_cols = [col for col in df.columns if '_pm10' in col]
        exclude_cols.extend(pm10_cols)
    
    # 특성 선택
    features = [c for c in df.columns if c not in exclude_cols]
    print(f"선택된 특성 수: {len(features)}")
    
    # 데이터 준비
    X = df[features].fillna(method="ffill").fillna(method="bfill")
    y = df[target]
    
    print(f"{target.upper()} 전체 행: {len(df):,}")
    print(f"{target.upper()} NaN 아닌 타깃 행: {y.notna().sum():,}")
    
    # 시계열 교차 검증 설정
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 모델 학습
    model = lgb.LGBMRegressor(
        objective="regression",
        num_leaves=63, 
        learning_rate=0.05, 
        n_estimators=500, 
        subsample=0.8, 
        colsample_bytree=0.8
    )
    
    for tr, va in tscv.split(X):
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
    
    # 모델 저장
    model.booster_.save_model(str(output_dir/f"lgbm_model_{target}.txt"))
    
    # 예측 및 평가
    pred = model.predict(X)
    
    # 타깃·예측 모두 유효한(=NaN 아님) 행만 남김
    valid = (~np.isnan(pred)) & (y.notna())
    y_valid = y[valid]
    pred_valid = pred[valid]
    
    # 지표 계산
    rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
    mae = mean_absolute_error(y_valid, pred_valid)
    r2 = r2_score(y_valid, pred_valid)
    
    metrics = {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}
    with open(output_dir/f"metrics_{target}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"{target.upper()} METRICS:", metrics)
    
    # SHAP 분석
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f'SHAP Summary for {target.upper()}')
    plt.tight_layout()
    plt.savefig(output_dir/f"shap_summary_{target}.png")
    plt.close()
    
    # 특성 중요도 시각화
    plt.figure(figsize=(12, 8))
    lgb.plot_importance(model, max_num_features=20)
    plt.title(f'{target.upper()} LightGBM Feature Importance')
    plt.tight_layout()
    plt.savefig(output_dir/f"feature_importance_{target}.png")
    plt.close()
    
    # 계절별 분석
    df['month'] = df['date'].dt.month
    spring_data = df[(df['month'] >= 3) & (df['month'] <= 5)].copy()
    summer_data = df[(df['month'] >= 6) & (df['month'] <= 8)].copy()
    winter_data = df[(df['month'] >= 12) | (df['month'] <= 2)].copy()
    
    seasons = {
        'spring': spring_data,
        'summer': summer_data,
        'winter': winter_data
    }
    
    for season_name, season_df in seasons.items():
        if len(season_df) > 0:
            X_season = season_df[features].fillna(method="ffill").fillna(method="bfill")
            y_season = season_df[target]
            
            # 모델 예측
            pred_season = model.predict(X_season)
            
            # NaN 제거
            valid = (~np.isnan(pred_season)) & (y_season.notna())
            y_valid = y_season[valid]
            pred_valid = pred_season[valid]
            
            # 메트릭 계산
            if len(y_valid) > 0:
                rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
                mae = mean_absolute_error(y_valid, pred_valid)
                r2 = r2_score(y_valid, pred_valid)
                
                season_metrics = {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}
                
                with open(output_dir/f"{season_name}_{target}_metrics.json", "w") as f:
                    json.dump(season_metrics, f, indent=2)
                
                # 계절별 SHAP 값
                season_shap_values = explainer.shap_values(X_season)
                plt.figure(figsize=(12, 8))
                shap.summary_plot(season_shap_values, X_season, show=False)
                plt.title(f'SHAP Summary for {season_name.capitalize()} - {target.upper()}')
                plt.tight_layout()
                plt.savefig(output_dir/f"{season_name}_{target}_shap_summary.png")
                plt.close()
    
    # 데이터셋 저장
    df.to_parquet(output_dir/f"final_dataset_{target}.parquet", index=False)
    
    # 국내 오염원과 국외 오염원의 영향 분석
    domestic_cols = [col for col in features if not (col.endswith('_pm10') or col.endswith('_pm25') or col == 'city')]
    foreign_cols = [col for col in features if col.endswith('_pm10') or col.endswith('_pm25')]
    
    # 국내 오염원만 사용한 모델
    if len(domestic_cols) > 0:
        X_domestic = df[domestic_cols].fillna(method="ffill").fillna(method="bfill")
        model_domestic = lgb.LGBMRegressor(
            objective="regression",
            num_leaves=63, 
            learning_rate=0.05, 
            n_estimators=500, 
            subsample=0.8, 
            colsample_bytree=0.8
        )
        
        for tr, va in tscv.split(X_domestic):
            model_domestic.fit(
                X_domestic.iloc[tr], y.iloc[tr],
                eval_set=[(X_domestic.iloc[va], y.iloc[va])],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
        
        pred_domestic = model_domestic.predict(X_domestic)
        valid = (~np.isnan(pred_domestic)) & (y.notna())
        y_valid = y[valid]
        pred_valid = pred_domestic[valid]
        
        if len(y_valid) > 0:
            rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
            mae = mean_absolute_error(y_valid, pred_valid)
            r2 = r2_score(y_valid, pred_valid)
            
            domestic_metrics = {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}
            with open(output_dir/f"domestic_only_{target}_metrics.json", "w") as f:
                json.dump(domestic_metrics, f, indent=2)
            
            print(f"{target.upper()} 국내 오염원만 사용 METRICS:", domestic_metrics)
    
    # 국외 오염원만 사용한 모델
    if len(foreign_cols) > 0:
        X_foreign = df[foreign_cols].fillna(method="ffill").fillna(method="bfill")
        model_foreign = lgb.LGBMRegressor(
            objective="regression",
            num_leaves=63, 
            learning_rate=0.05, 
            n_estimators=500, 
            subsample=0.8, 
            colsample_bytree=0.8
        )
        
        for tr, va in tscv.split(X_foreign):
            model_foreign.fit(
                X_foreign.iloc[tr], y.iloc[tr],
                eval_set=[(X_foreign.iloc[va], y.iloc[va])],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
        
        pred_foreign = model_foreign.predict(X_foreign)
        valid = (~np.isnan(pred_foreign)) & (y.notna())
        y_valid = y[valid]
        pred_valid = pred_foreign[valid]
        
        if len(y_valid) > 0:
            rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
            mae = mean_absolute_error(y_valid, pred_valid)
            r2 = r2_score(y_valid, pred_valid)
            
            foreign_metrics = {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}
            with open(output_dir/f"foreign_only_{target}_metrics.json", "w") as f:
                json.dump(foreign_metrics, f, indent=2)
            
            print(f"{target.upper()} 국외 오염원만 사용 METRICS:", foreign_metrics)
    
    # 국내외 오염원 영향 비교 시각화
    if len(domestic_cols) > 0 and len(foreign_cols) > 0:
        comparison = {
            '전체 특성': metrics['R2'],
            '국내 오염원만': domestic_metrics['R2'],
            '국외 오염원만': foreign_metrics['R2']
        }
        
        plt.figure(figsize=(10, 6))
        plt.bar(comparison.keys(), comparison.values())
        plt.title(f'{target.upper()} 국내외 오염원 영향 비교 (R²)')
        plt.ylabel('R² Score')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir/f"domestic_foreign_comparison_{target}.png")
        plt.close()
    
    print(f"===== {target.upper()} 회귀분석 완료 =====\n")
    return model, metrics

# ───── 7. 회귀분석 실행 ─────
# PM2.5 회귀분석
model_pm25, metrics_pm25 = run_regression_analysis(df_pm25, "pm25", OUT_PM25)

# PM10 회귀분석
model_pm10, metrics_pm10 = run_regression_analysis(df_pm10, "pm10", OUT_PM10)

# ───── 8. 결과 요약 ─────
print("\n===== 회귀분석 결과 요약 =====")
print("PM2.5 회귀분석 결과:", metrics_pm25)
print("PM10 회귀분석 결과:", metrics_pm10)
print("결과는 다음 폴더에 저장되었습니다:")
print(f"PM2.5 결과: {OUT_PM25}")
print(f"PM10 결과: {OUT_PM10}")
print("===== 분석 완료 =====")
