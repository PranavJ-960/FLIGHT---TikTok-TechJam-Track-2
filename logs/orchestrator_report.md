# Autonomous orchestrator run report

- Stop reason: **converged**
- Iterations run: 14 / 50 cap
- Wall-clock: 504.8s (budget 3600.0s)
- Baseline valid primary: 0.6015
- Best valid primary found: 0.6349 (delta +0.0334)
- Best iteration: #10 via blend (blend)
- Params: `{'lgb': 0.2, 'xgb': 0.5, 'catboost': 0.3}`
- Errors encountered and recovered from: 0

## Iteration log

- **[0]** LightGBM trial: {'learning_rate': 0.08, 'num_leaves': 127, 'min_data_in_leaf': 100, 'feature_fraction': 1.0, 'bagging_fraction': 0.6, 'lambda_l1': 0.0, 'lambda_l2': 10.0, 'max_depth': 12, 'min_gain_to_split': 0.0} -> primary=0.6226 (14.9s)
- **[1]** XGBoost trial: {'eta': 0.2, 'max_depth': 6, 'min_child_weight': 10, 'subsample': 1.0, 'colsample_bytree': 0.6, 'lambda': 1.0, 'alpha': 0.1, 'gamma': 0.1} -> primary=0.6280 (7.5s)
- **[2]** feature variant smooth=1.0 recent_window=10 -> primary=0.6272 (16.3s)
- **[3]** CatBoost trial: {'learning_rate': 0.15, 'depth': 7, 'l2_leaf_reg': 10.0, 'subsample': 0.8, 'rsm': 1.0, 'random_strength': 0.0} -> primary=0.6326 (98.2s)
- **[4]** XGBoost trial: {'eta': 0.15, 'max_depth': 4, 'min_child_weight': 20, 'subsample': 1.0, 'colsample_bytree': 0.5, 'lambda': 0.1, 'alpha': 0.1, 'gamma': 1.0} -> primary=0.6336 (16.1s)
- **[5]** XGBoost trial: {'eta': 0.03, 'max_depth': 8, 'min_child_weight': 5, 'subsample': 0.7, 'colsample_bytree': 0.7, 'lambda': 1.0, 'alpha': 0.0, 'gamma': 1.0} -> primary=0.6279 (32.6s)
- **[6]** feature variant smooth=20.0 recent_window=40 -> primary=0.6262 (15.1s)
- **[7]** CatBoost trial: {'learning_rate': 0.15, 'depth': 5, 'l2_leaf_reg': 5.0, 'subsample': 0.6, 'rsm': 0.6, 'random_strength': 5.0} -> primary=0.6323 (68.5s)
- **[8]** LightGBM trial: {'learning_rate': 0.12, 'num_leaves': 127, 'min_data_in_leaf': 100, 'feature_fraction': 0.7, 'bagging_fraction': 0.9, 'lambda_l1': 0.1, 'lambda_l2': 5.0, 'max_depth': 10, 'min_gain_to_split': 0.1} -> primary=0.6261 (5.3s)
- **[9]** CatBoost trial: {'learning_rate': 0.03, 'depth': 8, 'l2_leaf_reg': 3.0, 'subsample': 0.8, 'rsm': 0.6, 'random_strength': 1.0} -> primary=0.6289 (132.6s)
- **[10]** blend of ['lgb', 'xgb', 'catboost'] -> primary=0.6349 (7.6s)
- **[11]** CatBoost trial: {'learning_rate': 0.15, 'depth': 5, 'l2_leaf_reg': 3.0, 'subsample': 1.0, 'rsm': 0.8, 'random_strength': 1.0} -> primary=0.6328 (59.3s)
- **[12]** feature variant smooth=5.0 recent_window=40 -> primary=0.6265 (14.4s)
- **[13]** LightGBM trial: {'learning_rate': 0.12, 'num_leaves': 400, 'min_data_in_leaf': 10, 'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'lambda_l1': 1.0, 'lambda_l2': 10.0, 'max_depth': 5, 'min_gain_to_split': 0.01} -> primary=0.6296 (7.7s)
