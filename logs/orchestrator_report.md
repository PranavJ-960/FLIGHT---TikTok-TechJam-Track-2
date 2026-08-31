# Autonomous orchestrator run report

- Stop reason: **converged**
- Iterations run: 5 / 50 cap
- Wall-clock: 182.1s (budget 1800.0s)
- Baseline valid primary: 0.6015
- Best valid primary found: 0.6325 (delta +0.0310)
- Best iteration: #4 via blend (blend)
- Params: `{'lgb': 0.2, 'xgb': 0.4, 'catboost': 0.4}`
- Errors encountered and recovered from: 0

## Iteration log

- **[0]** LightGBM trial: {'learning_rate': 0.16, 'num_leaves': 400, 'min_data_in_leaf': 150, 'feature_fraction': 0.8, 'bagging_fraction': 1.0, 'lambda_l1': 1.0, 'lambda_l2': 10.0, 'max_depth': -1, 'min_gain_to_split': 0.01} -> primary=0.6229 (8.7s)
- **[1]** XGBoost trial: {'eta': 0.05, 'max_depth': 5, 'min_child_weight': 50, 'subsample': 1.0, 'colsample_bytree': 0.8, 'lambda': 0.0, 'alpha': 1.0, 'gamma': 1.0} -> primary=0.6303 (22.4s)
- **[2]** feature variant smooth=5.0 recent_window=40 -> primary=0.6265 (13.5s)
- **[3]** CatBoost trial: {'learning_rate': 0.05, 'depth': 8, 'l2_leaf_reg': 1.0, 'subsample': 0.8, 'rsm': 1.0, 'random_strength': 1.0} -> primary=0.6311 (121.5s)
- **[4]** blend of ['lgb', 'xgb', 'catboost'] -> primary=0.6325 (8.2s)
