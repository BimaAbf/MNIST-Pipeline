import numpy as np


class Metrics:

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        # Discover all class labels from both arrays
        labels = np.unique(np.concatenate([y_true, y_pred]))
        n = len(labels)
        label_to_idx = {label: i for i, label in enumerate(labels)}
        # Count co-occurrences: cm[true_class, predicted_class]
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[label_to_idx[t], label_to_idx[p]] += 1
        return cm

    @staticmethod
    def accuracy_score(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.sum(y_true == y_pred) / len(y_true)

    @staticmethod
    def _precision_recall_fscore_support(y_true, y_pred, average=None):
        """Compute per-class precision, recall, F1, and support, then aggregate.

        average=None  → return per-class arrays
        average="macro"    → unweighted mean across classes
        average="weighted" → support-weighted mean across classes
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        labels = np.unique(np.concatenate([y_true, y_pred]))

        precision_per = np.zeros(len(labels), dtype=float)
        recall_per = np.zeros(len(labels), dtype=float)
        f1_per = np.zeros(len(labels), dtype=float)
        support_per = np.zeros(len(labels), dtype=int)

        for i, label in enumerate(labels):
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))
            support_per[i] = tp + fn  # number of true instances for this class
            precision_per[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_per[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # Harmonic mean of precision and recall
            if (precision_per[i] + recall_per[i]) > 0:
                f1_per[i] = 2 * precision_per[i] * recall_per[i] / (precision_per[i] + recall_per[i])

        if average == "macro":
            # Simple average — treats all classes equally regardless of size
            return np.mean(precision_per), np.mean(recall_per), np.mean(f1_per), np.sum(support_per)
        elif average == "weighted":
            # Weight each class by its support (number of true instances)
            total = np.sum(support_per)
            if total == 0:
                return 0.0, 0.0, 0.0, 0
            w = support_per / total
            return np.dot(w, precision_per), np.dot(w, recall_per), np.dot(w, f1_per), total
        else:
            return precision_per, recall_per, f1_per, support_per

    @staticmethod
    def classification_report(y_true, y_pred, digits=2):
        """Build a text classification report with per-class and aggregate metrics."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        target_names = [str(l) for l in labels]

        # Per-class metrics
        p, r, f, s = Metrics._precision_recall_fscore_support(y_true, y_pred, average=None)

        # Format columns with dynamic width based on longest class name
        col_width = max(len(n) for n in target_names)
        fmt = f"{{:>{col_width}s}}" + f"  {{:>9.{digits}f}}" * 3 + "  {:>9d}\n"
        header_fmt = f"{{:>{col_width}s}}" + "  {:>9s}" * 3 + "  {:>9s}\n"

        lines = "\n"
        lines += header_fmt.format("", "precision", "recall", "f1-score", "support")
        lines += "\n"
        for i, name in enumerate(target_names):
            lines += fmt.format(name, p[i], r[i], f[i], int(s[i]))
        lines += "\n"

        # Overall accuracy row
        total_support = int(np.sum(s))
        acc = Metrics.accuracy_score(y_true, y_pred)
        acc_fmt = f"{{:>{col_width}s}}" + f"  {{:>9s}}" * 2 + f"  {{:>9.{digits}f}}" + "  {:>9d}\n"
        lines += acc_fmt.format("accuracy", "", "", acc, total_support)

        # Macro and weighted average rows
        for avg_name, avg_type in [("macro avg", "macro"), ("weighted avg", "weighted")]:
            pa, ra, fa, _ = Metrics._precision_recall_fscore_support(y_true, y_pred, average=avg_type)
            lines += fmt.format(avg_name, pa, ra, fa, total_support)
        lines += "\n"
        return lines

    @staticmethod
    def learning_curve(estimator, X, y, train_sizes, cv=3):
        """Evaluate model performance across increasing training set sizes using k-fold CV.

        Returns (actual_sizes, train_scores, val_scores) where scores have shape
        (len(train_sizes), cv).
        """
        n = len(X)
        indices = np.arange(n)
        rng = np.random.RandomState(42)
        rng.shuffle(indices)

        # Split indices into cv equal folds
        fold_size = n // cv
        folds = []
        for i in range(cv):
            start = i * fold_size
            end = start + fold_size if i < cv - 1 else n
            folds.append(indices[start:end])

        train_scores = np.zeros((len(train_sizes), cv))
        val_scores = np.zeros((len(train_sizes), cv))
        actual_sizes = np.zeros(len(train_sizes), dtype=int)

        for fold_i in range(cv):
            # Use fold_i as validation, rest as training
            val_idx = folds[fold_i]
            train_idx = np.concatenate([folds[j] for j in range(cv) if j != fold_i])

            for size_i, frac in enumerate(train_sizes):
                # Take a fraction of the training fold
                n_train = max(10, int(len(train_idx) * frac))
                subset = train_idx[:n_train]
                actual_sizes[size_i] = n_train

                # Create a fresh estimator with the same hyperparameters
                est = estimator.__class__(C=estimator.C, penalty=estimator.penalty)
                est.fit(X[subset], y[subset])

                train_scores[size_i, fold_i] = Metrics.accuracy_score(y[subset], est.predict(X[subset]))
                val_scores[size_i, fold_i] = Metrics.accuracy_score(y[val_idx], est.predict(X[val_idx]))

        return actual_sizes, train_scores, val_scores
