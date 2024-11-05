from typing import Any

import numpy
import polars
from datasets import Dataset, load_dataset
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

dataset = load_dataset(
	"csv", data_dir="data", data_files={"train": "train.csv", "test": "test.csv"}, keep_in_memory=True, num_proc=8
)

index_columns: list = ["data_ID", "player_ID"]
label_columns: list = ["gender", "hold racket handed", "play years", "level"]
data_columns: list = list(set(dataset["train"].column_names).difference(set(index_columns + label_columns)))

scaler = MinMaxScaler()

train_dataset: polars.DataFrame = dataset["train"].to_polars()
train_data: numpy.ndarray = scaler.fit_transform(train_dataset[data_columns].to_numpy())
train_label: dict[str, numpy.ndarray] = {label: train_dataset[label].to_numpy() for label in label_columns}

hyperparameter_grid: dict[str, list[Any]] = {
	"n_neighbors": [3, 5, 7, 9, 11],  # 鄰居數量
	"weights": ["uniform", "distance"],  # 權重類型
	"algorithm": ["auto", "ball_tree", "kd_tree", "brute"],  # 演算法類型
	"leaf_size": [20, 30, 40, 50],  # 葉節點大小
	"p": [1, 2],  # 距離度量的參數 (1 為曼哈頓距離, 2 為歐幾里得距離)
}

best_classifiers: dict = {}
for label in label_columns:
	print(f"Tune classifier for {label}")
	classifier = KNeighborsClassifier(n_jobs=-1)

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

test_result.to_csv("../dist/result_kn.csv", num_proc=8)
