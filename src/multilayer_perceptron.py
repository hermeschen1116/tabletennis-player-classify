from typing import Any

import numpy
import polars
from datasets import Dataset, load_dataset
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

dataset = load_dataset(
	"csv", data_dir="data", data_files={"train": "train.csv", "test": "test.csv"}, keep_in_memory=True, num_proc=8
)

index_columns: list = ["data_ID", "player_ID"]
label_columns: list = ["gender", "hold racket handed", "play years", "level"]
data_columns: list = list(set(dataset["train"].column_names).difference(set(index_columns + label_columns)))

scaler = MinMaxScaler()
scaler.fit(dataset["train"].to_polars()[data_columns].to_numpy())

train_dataset: polars.DataFrame = dataset["train"].to_polars()
train_data: numpy.ndarray = scaler.transform(train_dataset[data_columns].to_numpy())
train_label: dict[str, numpy.ndarray] = {label: train_dataset[label].to_numpy() for label in label_columns}

hyperparameter_grid: dict[str, list[Any]] = {
	"hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],  # 隱藏層層數及神經元數
	"activation": ["identity", "logistic", "tanh", "relu"],  # 激活函數
	"solver": ["lbfgs", "sgd", "adam"],  # 權重優化器
	"alpha": [0.0001, 0.001, 0.01, 0.1],  # L2 正則化係數
	"learning_rate": ["constant", "invscaling", "adaptive"],  # 學習率調整方式
	"learning_rate_init": [0.001, 0.01, 0.1],  # 初始學習率
	"max_iter": [200, 300, 500],  # 最大迭代次數
	"tol": [1e-4, 1e-3, 1e-2],  # 收斂公差
	"momentum": [0.9, 0.95, 0.99],  # 動量（僅對 solver='sgd' 有效）
	"nesterovs_momentum": [True, False],  # 是否使用 Nesterov 動量（僅對 solver='sgd' 有效）
	"early_stopping": [True, False],  # 是否使用早期停止
	"validation_fraction": [0.1, 0.2, 0.3],  # 用於早期停止的驗證集比例
	"beta_1": [0.9, 0.95, 0.99],  # Adam 優化的第一個動量參數（僅對 solver='adam' 有效）
	"beta_2": [0.999, 0.995, 0.99],  # Adam 優化的第二個動量參數（僅對 solver='adam' 有效）
	"epsilon": [1e-8, 1e-7, 1e-6],  # Adam 優化中的數值穩定項（僅對 solver='adam' 有效）
}

best_classifiers: dict = {}
for label in label_columns:
	print(f"Tune classifier for {label}")
	classifier = MLPClassifier(random_state=37)

	scoring: str = "roc_auc" if label in ["gender", "hold racket handed"] else "roc_auc_ovr"
	tuner = RandomizedSearchCV(
		classifier, hyperparameter_grid, n_iter=100, cv=5, scoring=scoring, n_jobs=-1, verbose=10
	)

	tuner.fit(train_data, train_label[label])

	best_classifiers[label] = tuner.best_estimator_

test_data: polars.DataFrame = scaler.transform(dataset["test"].select_columns(data_columns).to_polars().to_numpy())

predictions: dict = {"data_ID": dataset["test"]["data_ID"]}
for label in label_columns:
	predictions[label] = best_classifiers[label].predict(test_data).tolist()

test_result = Dataset.from_dict(predictions)

test_result = test_result.map(
	lambda samples: {
		"play years_0": [1 if sample == 0 else 0 for sample in samples],
		"play years_1": [1 if sample == 1 else 0 for sample in samples],
		"play years_2": [1 if sample == 2 else 0 for sample in samples],
	},
	input_columns=["play years"],
	remove_columns=["play years"],
	batched=True,
	num_proc=8,
)

test_result = test_result.map(
	lambda samples: {
		"level_0": [1 if sample == 0 else 0 for sample in samples],
		"level_1": [1 if sample == 1 else 0 for sample in samples],
		"level_2": [1 if sample == 2 else 0 for sample in samples],
	},
	input_columns=["level"],
	remove_columns=["level"],
	batched=True,
	num_proc=8,
)


test_result.to_csv("../dist/result_mp.csv", num_proc=8)
