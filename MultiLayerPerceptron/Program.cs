using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;

// Regression with the custom 3-layer MLP. The engine files (BackNeural.cs, InputN.cs, HiddenN.cs,
// OuputN.cs) are used exactly as provided; everything else lives in this file.
namespace Backprop
{
    /// <summary>
    /// Regression with the custom 3-layer MLP (Backprop.NeuralNet).
    /// Pipeline: self-test -> load CSV -> split -> grid search (LOOCV) -> train with
    /// loss log -> evaluate -> leave-one-out check on all rows -> sample prediction.
    /// Usage:  dotnet run [-- path/to/dataset.csv]
    /// </summary>
    public class Program
    {
        private const double TrainRatio = 0.75;
        private const int SplitSeed = 42;
        private const int FinalLogLines = 20;   // number of loss lines printed during final training

        public static int Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
            string path = args.Length > 0 ? args[0] : FindDataset();

            Banner("1. Engine verification (gradient check of the unmodified engine)");
            if (!EngineSelfTest.Run()) return 1;

            Banner("2. Data loading & description");
            int skipped;
            Dataset data = CsvLoader.Load(path, out skipped);
            DescribeData(data, path, skipped);

            Banner("3. Train / test split");
            int[] train, test;
            DataSplitter.Split(data.Count, TrainRatio, SplitSeed, out train, out test);
            Console.WriteLine("  Shuffled split (seed {0}): {1} training rows, {2} test rows", SplitSeed, train.Length, test.Length);
            Console.WriteLine("  Train rows: {0}", string.Join(", ", Array.ConvertAll(train, i => (i + 1).ToString())));
            Console.WriteLine("  Test rows : {0}", string.Join(", ", Array.ConvertAll(test, i => (i + 1).ToString())));
            Console.WriteLine("  Scalers are fitted on training rows only (features -> [0,1], target -> [{0},{1}]).",
                RegressionModel.TargetLow, RegressionModel.TargetHigh);

            Banner("4. Hyperparameter tuning (hidden nodes x epochs, leave-one-out CV on training rows)");
            Console.WriteLine("  Learning rates are fixed by the engine: output layer 0.2, input layer 0.15.");
            int total = HyperparameterTuner.HiddenSizes.Length * HyperparameterTuner.EpochCounts.Length;
            Console.WriteLine("  {0} configurations x {1} seeds x {2} folds ...", total, HyperparameterTuner.Seeds.Length, train.Length);
            List<TuningResult> results = HyperparameterTuner.Run(data, train);
            PrintTuning(results);
            TrainingConfig best = results[0].Config;

            Banner("5. Final training log (best configuration)");
            Console.WriteLine("  Config: {0}", best);
            best.LogInterval = Math.Max(1, best.Epochs / FinalLogLines);
            RegressionModel model = new RegressionModel();
            model.Fit(data, train, best);

            Banner("6. Evaluation");
            double[] trainActual = Select(data.Y, train), testActual = Select(data.Y, test);
            double[] trainPred = model.Predict(data, train), testPred = model.Predict(data, test);
            RegressionMetrics mTrain = RegressionMetrics.Compute(trainActual, trainPred);
            RegressionMetrics mTest = RegressionMetrics.Compute(testActual, testPred);
            // baseline: always predict the training mean
            double mean = 0; foreach (double v in trainActual) mean += v; mean /= trainActual.Length;
            double[] baselinePred = new double[testActual.Length];
            for (int i = 0; i < baselinePred.Length; i++) baselinePred[i] = mean;
            RegressionMetrics mBase = RegressionMetrics.Compute(testActual, baselinePred);

            Console.WriteLine("  Train    : {0}", mTrain);
            Console.WriteLine("  Test     : {0}", mTest);
            Console.WriteLine("  Baseline : {0}   (always predicts the training mean {1:F3})", mBase, mean);
            Console.WriteLine();
            PrintPredictions("Test set predictions", data, test, testPred);
            PrintPredictions("Training set predictions", data, train, trainPred);

            Banner("7. Leave-one-out cross-validation on ALL rows (more reliable for tiny data)");
            double[] looActual = new double[data.Count], looPred = new double[data.Count];
            for (int leave = 0; leave < data.Count; leave++)
            {
                List<int> fold = new List<int>();
                for (int i = 0; i < data.Count; i++) if (i != leave) fold.Add(i);
                TrainingConfig cfg = best;
                cfg.LogInterval = 0;
                RegressionModel m = new RegressionModel();
                m.Fit(data, fold.ToArray(), cfg);
                looActual[leave] = data.Y[leave];
                looPred[leave] = m.Predict(data.X[leave]);
            }
            RegressionMetrics mLoo = RegressionMetrics.Compute(looActual, looPred);
            Console.WriteLine("  LOOCV    : {0}", mLoo);
            Console.WriteLine();

            Banner("8. Summary");
            Summarise(mTrain, mTest, mBase, mLoo, data, train, test);

            Banner("9. Example real-world prediction");
            double[] example = new double[data.FeatureCount];
            for (int f = 0; f < example.Length; f++) example[f] = Median(data, f);
            double prediction = model.Predict(example);
            Console.Write("  Input ");
            for (int f = 0; f < example.Length; f++) Console.Write("{0}={1:F2}  ", data.FeatureNames[f], example[f]);
            Console.WriteLine("-> predicted {0}: {1:F3}", data.TargetName, prediction);
            return 0;
        }

        // ------------------------------------------------------------------ helpers
        private static string FindDataset()
        {
            string[] candidates = { "dataset.csv", Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "dataset.csv") };
            foreach (string c in candidates) if (File.Exists(c)) return c;
            throw new FileNotFoundException("dataset.csv not found. Pass the path as the first argument.");
        }

        private static void Banner(string title)
        {
            Console.WriteLine("==============================================================");
            Console.WriteLine(" " + title);
            Console.WriteLine("==============================================================");
        }

        private static double[] Select(List<double> values, int[] idx)
        {
            double[] r = new double[idx.Length];
            for (int i = 0; i < idx.Length; i++) r[i] = values[idx[i]];
            return r;
        }

        private static double Median(Dataset d, int feature)
        {
            double[] v = new double[d.Count];
            for (int i = 0; i < v.Length; i++) v[i] = d.X[i][feature];
            Array.Sort(v);
            return v.Length % 2 == 1 ? v[v.Length / 2] : (v[v.Length / 2 - 1] + v[v.Length / 2]) / 2.0;
        }

        private static void DescribeData(Dataset d, string path, int skipped)
        {
            Console.WriteLine("  File    : {0}", path);
            Console.WriteLine("  Rows    : {0} valid ({1} dropped for missing/malformed values)", d.Count, skipped);
            Console.WriteLine("  Task    : regression, predict '{0}' from {1} feature(s)", d.TargetName, d.FeatureCount);
            Console.WriteLine();
            Console.WriteLine("  {0,-16} {1,8} {2,8} {3,8}   {4}", "Column", "Min", "Max", "Mean", "Corr. with target");
            for (int f = 0; f <= d.FeatureCount; f++)
            {
                double[] col = new double[d.Count];
                for (int i = 0; i < d.Count; i++) col[i] = f < d.FeatureCount ? d.X[i][f] : d.Y[i];
                double mn = double.MaxValue, mx = double.MinValue, avg = 0;
                foreach (double v in col) { mn = Math.Min(mn, v); mx = Math.Max(mx, v); avg += v; }
                avg /= col.Length;
                string name = f < d.FeatureCount ? d.FeatureNames[f] : d.TargetName + " (target)";
                string corr = f < d.FeatureCount ? Correlation(col, d.Y.ToArray()).ToString("F3") : "-";
                Console.WriteLine("  {0,-16} {1,8:F3} {2,8:F3} {3,8:F3}   {4}", name, mn, mx, avg, corr);
            }
            if (d.Count < 30)
                Console.WriteLine("\n  WARNING: only {0} rows - metrics will be noisy; add more data for trustworthy results.", d.Count);
        }

        private static double Correlation(double[] a, double[] b)
        {
            double ma = 0, mb = 0;
            for (int i = 0; i < a.Length; i++) { ma += a[i]; mb += b[i]; }
            ma /= a.Length; mb /= b.Length;
            double sab = 0, saa = 0, sbb = 0;
            for (int i = 0; i < a.Length; i++)
            {
                sab += (a[i] - ma) * (b[i] - mb);
                saa += (a[i] - ma) * (a[i] - ma);
                sbb += (b[i] - mb) * (b[i] - mb);
            }
            return saa == 0 || sbb == 0 ? 0 : sab / Math.Sqrt(saa * sbb);
        }

        private static void PrintTuning(List<TuningResult> results)
        {
            Console.WriteLine();
            Console.WriteLine("  All configurations, best first (lower LOOCV RMSE is better):");
            Console.WriteLine("  {0,4} | {1,6} | {2,6} | {3,9} | {4,9}", "Rank", "Hidden", "Epochs", "CV RMSE", "CV MAE");
            for (int i = 0; i < results.Count; i++) PrintTuningRow(i + 1, results[i]);
            Console.WriteLine();
            Console.WriteLine("  Selected: {0}  (CV RMSE {1:F4})", results[0].Config, results[0].CvRmse);
        }

        private static void PrintTuningRow(int rank, TuningResult r)
        {
            Console.WriteLine("  {0,4} | {1,6} | {2,6} | {3,9:F4} | {4,9:F4}", rank,
                r.Config.HiddenNodes, r.Config.Epochs, r.CvRmse, r.CvMae);
        }

        private static void PrintPredictions(string title, Dataset d, int[] idx, double[] pred)
        {
            Console.WriteLine("  {0}:", title);
            Console.WriteLine("  {0,4} | {1,10} | {2,10} | {3,10} | {4,10}", "Row", "Actual", "Predicted", "Error", "Abs error");
            for (int i = 0; i < idx.Length; i++)
            {
                double actual = d.Y[idx[i]];
                Console.WriteLine("  {0,4} | {1,10:F4} | {2,10:F4} | {3,10:F4} | {4,10:F4}", idx[i] + 1, actual, pred[i], pred[i] - actual, Math.Abs(pred[i] - actual));
            }
            Console.WriteLine();
        }

        private static int CountOutsideTrainingRange(Dataset d, int[] train, int[] test)
        {
            int cols = d.FeatureCount + 1;
            double[] mn = new double[cols], mx = new double[cols];
            for (int c = 0; c < cols; c++) { mn[c] = double.MaxValue; mx[c] = double.MinValue; }
            foreach (int i in train)
                for (int c = 0; c < cols; c++)
                {
                    double v = c < d.FeatureCount ? d.X[i][c] : d.Y[i];
                    mn[c] = Math.Min(mn[c], v); mx[c] = Math.Max(mx[c], v);
                }
            int count = 0;
            foreach (int i in test)
            {
                bool outside = false;
                for (int c = 0; c < cols; c++)
                {
                    double v = c < d.FeatureCount ? d.X[i][c] : d.Y[i];
                    if (v < mn[c] || v > mx[c]) outside = true;
                }
                if (outside) count++;
            }
            return count;
        }

        private static void Summarise(RegressionMetrics train, RegressionMetrics test, RegressionMetrics baseline, RegressionMetrics loo, Dataset d, int[] trainIdx, int[] testIdx)
        {
            Console.WriteLine("  * Test RMSE {0:F4} vs. mean-predictor RMSE {1:F4}: the network is {2:F1}x more accurate than guessing the average.",
                test.Rmse, baseline.Rmse, baseline.Rmse / Math.Max(test.Rmse, 1e-12));
            Console.WriteLine("  * Test RMSE / Train RMSE = {0:F2}: {1}", test.Rmse / Math.Max(train.Rmse, 1e-12),
                test.Rmse > 3 * train.Rmse ? "possible overfitting (test error much larger than train error)." : "no strong sign of overfitting.");
            Console.WriteLine("  * LOOCV R2 = {0:F3} over all {1} rows (each row predicted by a model that never saw it).", loo.R2, d.Count);
            int outside = CountOutsideTrainingRange(d, trainIdx, testIdx);
            if (outside > 0)
                Console.WriteLine("  * {0} of {1} test row(s) lie OUTSIDE the training range of the target or a feature (extrapolation). A sigmoid network cannot extrapolate, so the test error is inflated by the split, not only by the model.", outside, testIdx.Length);
            Console.WriteLine("  * Caveat: with {0} rows the test set has only {1} row(s); treat these numbers as indicative, not conclusive.", d.Count, test.Count);
        }
    }

    /// <summary>A numeric tabular dataset: feature rows plus one target value per row.</summary>
    public class Dataset
    {
        public string[] FeatureNames;
        public string TargetName;
        public List<double[]> X = new List<double[]>();
        public List<double> Y = new List<double>();
        public int Count { get { return X.Count; } }
        public int FeatureCount { get { return FeatureNames.Length; } }
    }

    /// <summary>Reads a CSV file: all columns but the last are features, the last is the target.</summary>
    public static class CsvLoader
    {
        public static Dataset Load(string path, out int skippedRows)
        {
            skippedRows = 0;
            string[] lines = File.ReadAllLines(path);
            Dataset data = new Dataset();
            bool headerRead = false;
            int columns = -1;

            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i].Trim();
                if (line.Length == 0) continue;
                string[] cells = line.Split(',');

                if (!headerRead)
                {
                    columns = cells.Length;
                    if (columns < 2)
                        throw new InvalidDataException("CSV needs at least 1 feature column and 1 target column.");
                    double ignored;
                    bool firstRowIsNumeric = double.TryParse(cells[0].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out ignored);
                    headerRead = true;
                    data.FeatureNames = new string[columns - 1];
                    if (!firstRowIsNumeric)
                    {
                        for (int c = 0; c < columns - 1; c++) data.FeatureNames[c] = cells[c].Trim();
                        data.TargetName = cells[columns - 1].Trim();
                        continue; // header consumed
                    }
                    for (int c = 0; c < columns - 1; c++) data.FeatureNames[c] = "Feature" + (c + 1);
                    data.TargetName = "Target";
                    // no header: fall through and parse this line as data
                }

                // malformed / missing values -> row is dropped and counted
                double[] values = new double[columns];
                bool ok = cells.Length == columns;
                for (int c = 0; ok && c < columns; c++)
                {
                    ok = double.TryParse(cells[c].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out values[c])
                        && !double.IsNaN(values[c]) && !double.IsInfinity(values[c]);
                }
                if (!ok) { skippedRows++; continue; }

                double[] features = new double[columns - 1];
                Array.Copy(values, features, columns - 1);
                data.X.Add(features);
                data.Y.Add(values[columns - 1]);
            }
            if (data.Count == 0) throw new InvalidDataException("No valid rows found in " + path);
            return data;
        }
    }

    /// <summary>
    /// Min-max scaler: maps each column from [min,max] (learned on the TRAINING
    /// data only, to avoid leakage) into [lo,hi].
    /// </summary>
    public class MinMaxScaler
    {
        private double[] min, max;
        private readonly double lo, hi;

        public MinMaxScaler(double lo, double hi) { this.lo = lo; this.hi = hi; }

        public void Fit(List<double[]> rows)
        {
            int n = rows[0].Length;
            min = new double[n]; max = new double[n];
            for (int c = 0; c < n; c++) { min[c] = double.MaxValue; max[c] = double.MinValue; }
            foreach (double[] r in rows)
                for (int c = 0; c < n; c++)
                {
                    if (r[c] < min[c]) min[c] = r[c];
                    if (r[c] > max[c]) max[c] = r[c];
                }
        }
        public void Fit(List<double> column)
        {
            List<double[]> rows = new List<double[]>();
            foreach (double v in column) rows.Add(new double[] { v });
            Fit(rows);
        }

        public double Transform(double value, int column)
        {
            double range = max[column] - min[column];
            if (range == 0) return (lo + hi) / 2.0; // constant column
            return lo + (value - min[column]) / range * (hi - lo);
        }
        public double[] Transform(double[] row)
        {
            double[] result = new double[row.Length];
            for (int c = 0; c < row.Length; c++) result[c] = Transform(row[c], c);
            return result;
        }
        public double Inverse(double scaled, int column)
        {
            double range = max[column] - min[column];
            if (range == 0) return min[column];
            return min[column] + (scaled - lo) / (hi - lo) * range;
        }
    }

    public static class DataSplitter
    {
        /// <summary>Shuffles row indices (seeded Fisher-Yates).</summary>
        public static int[] ShuffledIndices(int n, int seed)
        {
            int[] idx = new int[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            Random rng = new Random(seed);
            for (int i = n - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
            return idx;
        }

        /// <summary>Random train/test split. At least one row always goes to each side.</summary>
        public static void Split(int n, double trainRatio, int seed, out int[] train, out int[] test)
        {
            int[] idx = ShuffledIndices(n, seed);
            int trainCount = (int)Math.Round(n * trainRatio);
            if (trainCount < 1) trainCount = 1;
            if (trainCount > n - 1) trainCount = n - 1;
            train = new int[trainCount];
            test = new int[n - trainCount];
            Array.Copy(idx, 0, train, 0, trainCount);
            Array.Copy(idx, trainCount, test, 0, n - trainCount);
        }
    }

    /// <summary>All hyperparameters of one training run.</summary>
    public class TrainingConfig
    {
        public int HiddenNodes = 4;
        public int Epochs = 2000;
        public int Seed = 1;                      // seeds the per-epoch sample shuffling
        public int LogInterval = 0;               // 0 = silent, otherwise print loss every N epochs

        public override string ToString()
        {
            return string.Format("hidden={0,-2} epochs={1}", HiddenNodes, Epochs);
        }
    }

    /// <summary>
    /// Wraps the Backprop.NeuralNet engine for regression: scales features and
    /// target with min-max scalers fitted on the training rows, trains with
    /// shuffled online back-propagation and predicts in the ORIGINAL target units.
    /// </summary>
    public class RegressionModel
    {
        // Features -> [0,1]. Target -> [0.1,0.9]: the output neuron is a sigmoid, which can
        // only reach 0 or 1 asymptotically, so targets at the very edge would saturate it.
        public const double FeatureLow = 0.0, FeatureHigh = 1.0;
        public const double TargetLow = 0.1, TargetHigh = 0.9;

        private MinMaxScaler featureScaler = new MinMaxScaler(FeatureLow, FeatureHigh);
        private MinMaxScaler targetScaler = new MinMaxScaler(TargetLow, TargetHigh);
        private NeuralNet net;

        public void Fit(Dataset data, int[] trainIndices, TrainingConfig cfg)
        {
            // 1. fit scalers on training rows only
            List<double[]> rawX = new List<double[]>();
            List<double> rawY = new List<double>();
            foreach (int i in trainIndices) { rawX.Add(data.X[i]); rawY.Add(data.Y[i]); }
            featureScaler.Fit(rawX);
            targetScaler.Fit(rawY);

            double[][] x = new double[rawX.Count][];
            double[] y = new double[rawY.Count];
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = featureScaler.Transform(rawX[i]);
                y[i] = targetScaler.Transform(rawY[i], 0);
            }

            // 2. build the network
            // NOTE: the engine draws its initial weights from its own unseeded Random,
            // so every new network starts from different random weights.
            net = new NeuralNet(data.FeatureCount, cfg.HiddenNodes, 1);

            // 3. training loop
            Random rng = new Random(cfg.Seed);
            int[] order = new int[x.Length];
            for (int i = 0; i < order.Length; i++) order[i] = i;
            double previousLogged = double.NaN;

            if (cfg.LogInterval > 0)
            {
                Console.WriteLine("  {0,8} | {1,12} | {2,14} | {3,12}", "Epoch", "Train MSE", "RMSE (orig.)", "Change");
                Console.WriteLine("  ---------+--------------+----------------+-------------");
            }
            for (int epoch = 1; epoch <= cfg.Epochs; epoch++)
            {
                // reshuffle sample order every epoch (Fisher-Yates)
                for (int i = order.Length - 1; i > 0; i--)
                {
                    int j = rng.Next(i + 1);
                    int t = order[i]; order[i] = order[j]; order[j] = t;
                }
                foreach (int s in order)
                {
                    for (int f = 0; f < x[s].Length; f++) net.setInputs(f, x[s][f]);
                    net.setDesiredOutput(0, y[s]);
                    net.learn();
                }

                bool logNow = cfg.LogInterval > 0 && (epoch == 1 || epoch % cfg.LogInterval == 0 || epoch == cfg.Epochs);
                if (logNow)
                {
                    double mse = ScaledMse(x, y);
                    double rmseOriginal = Math.Sqrt(mse) / (TargetHigh - TargetLow) * (rawYRange(rawY));
                    string change = double.IsNaN(previousLogged) ? "-" : (mse - previousLogged).ToString("+0.000000;-0.000000");
                    Console.WriteLine("  {0,8} | {1,12:F6} | {2,14:F5} | {3,12}", epoch, mse, rmseOriginal, change);
                    previousLogged = mse;
                }
            }
        }

        private static double rawYRange(List<double> y)
        {
            double mn = double.MaxValue, mx = double.MinValue;
            foreach (double v in y) { if (v < mn) mn = v; if (v > mx) mx = v; }
            return (mx - mn) == 0 ? 1 : (mx - mn);
        }

        /// <summary>Mean squared error in scaled units - the quantity back-propagation minimises.</summary>
        private double ScaledMse(double[][] x, double[] y)
        {
            double sum = 0;
            for (int s = 0; s < x.Length; s++)
            {
                for (int f = 0; f < x[s].Length; f++) net.setInputs(f, x[s][f]);
                net.run();
                double e = y[s] - net.getOuputData(0);
                sum += e * e;
            }
            return sum / x.Length;
        }

        /// <summary>Predict one raw (unscaled) feature row; returns the value in original target units.</summary>
        public double Predict(double[] rawFeatures)
        {
            for (int f = 0; f < rawFeatures.Length; f++)
                net.setInputs(f, featureScaler.Transform(rawFeatures[f], f));
            net.run();
            return targetScaler.Inverse(net.getOuputData(0), 0);
        }

        public double[] Predict(Dataset data, int[] indices)
        {
            double[] result = new double[indices.Length];
            for (int i = 0; i < indices.Length; i++) result[i] = Predict(data.X[indices[i]]);
            return result;
        }
    }

    /// <summary>Standard regression metrics, computed in the original target units.</summary>
    public class RegressionMetrics
    {
        public int Count;
        public double Mse, Rmse, Mae, R2;

        public static RegressionMetrics Compute(double[] actual, double[] predicted)
        {
            int n = actual.Length;
            double mean = 0;
            for (int i = 0; i < n; i++) mean += actual[i];
            mean /= n;

            double sse = 0, sae = 0, sst = 0;
            for (int i = 0; i < n; i++)
            {
                double e = actual[i] - predicted[i];
                sse += e * e;
                sae += Math.Abs(e);
                sst += (actual[i] - mean) * (actual[i] - mean);
            }
            RegressionMetrics m = new RegressionMetrics();
            m.Count = n;
            m.Mse = sse / n;
            m.Rmse = Math.Sqrt(m.Mse);
            m.Mae = sae / n;
            m.R2 = sst > 0 ? 1.0 - sse / sst : double.NaN; // undefined when all actuals are equal
            return m;
        }

        public override string ToString()
        {
            return string.Format("n={0}  MSE={1:F5}  RMSE={2:F4}  MAE={3:F4}  R2={4:F4}", Count, Mse, Rmse, Mae, R2);
        }
    }

    public class TuningResult
    {
        public TrainingConfig Config;
        public double CvRmse;   // leave-one-out RMSE (original units), pooled over seeds
        public double CvMae;
    }

    /// <summary>
    /// Grid search over hidden size and epochs (the engine's learning rates are compile-time constants and cannot be varied). With only a handful of
    /// rows a single validation split would be extremely noisy, so every candidate is
    /// scored by leave-one-out cross-validation on the TRAINING rows (the test rows are
    /// never touched) and averaged over several random initialisations (3 repeats).
    /// </summary>
    public static class HyperparameterTuner
    {
        public static readonly int[] HiddenSizes = { 2, 4, 8, 16 };
        public static readonly int[] EpochCounts = { 500, 2000, 5000 };
        public static readonly int[] Seeds = { 1, 2, 3 };

        public static List<TuningResult> Run(Dataset data, int[] trainIndices)
        {
            List<TuningResult> results = new List<TuningResult>();
            foreach (int hidden in HiddenSizes)
                foreach (int epochs in EpochCounts)
                {
                    TrainingConfig cfg = new TrainingConfig();
                    cfg.HiddenNodes = hidden;
                    cfg.Epochs = epochs;

                    double sse = 0, sae = 0; int count = 0;
                    foreach (int seed in Seeds)
                    {
                        cfg.Seed = seed;
                        for (int leaveOut = 0; leaveOut < trainIndices.Length; leaveOut++)
                        {
                            int[] fold = new int[trainIndices.Length - 1];
                            for (int i = 0, k = 0; i < trainIndices.Length; i++)
                                if (i != leaveOut) fold[k++] = trainIndices[i];

                            RegressionModel model = new RegressionModel();
                            model.Fit(data, fold, cfg);
                            double e = data.Y[trainIndices[leaveOut]] - model.Predict(data.X[trainIndices[leaveOut]]);
                            sse += e * e; sae += Math.Abs(e); count++;
                        }
                    }
                    TuningResult r = new TuningResult();
                    r.Config = cfg;
                    r.CvRmse = Math.Sqrt(sse / count);
                    r.CvMae = sae / count;
                    results.Add(r);
                }
            results.Sort(delegate (TuningResult a, TuningResult b) { return a.CvRmse.CompareTo(b.CvRmse); });
            return results;
        }
    }

    /// <summary>
    /// Verifies the back-propagation maths of the UNMODIFIED engine. The engine exposes no
    /// weight accessors on NeuralNet, so weights are read/written through its own
    /// saveWeights()/loadWeights() file format. The analytic gradient implied by one learn()
    /// step is compared with a central finite-difference gradient of  L = 1/2 SUM_k (t_k - o_k)^2.
    /// </summary>
    public static class EngineSelfTest
    {
        // These mirror the private constants LRPIN / LRPOUT in BackNeural.cs.
        private const double LrInput = 0.15, LrOutput = 0.2;

        public static bool Run()
        {
            string file = Path.GetTempFileName();
            try
            {
                bool ok = CheckForwardPass(file);
                ok &= CheckGradients(file);
                Console.WriteLine(ok ? "  Self-test PASSED" : "  Self-test FAILED");
                Console.WriteLine();
                return ok;
            }
            finally { File.Delete(file); }
        }

        // file layout written by saveWeights: input->hidden [i][h], hidden->output [h][o], hidden biases, output biases
        private static double[] ReadParams(NeuralNet net, string file)
        {
            net.saveWeights(file);
            string[] lines = File.ReadAllLines(file);
            double[] w = new double[lines.Length];
            for (int i = 0; i < w.Length; i++) w[i] = double.Parse(lines[i], CultureInfo.InvariantCulture);
            return w;
        }

        private static void WriteParams(NeuralNet net, string file, double[] w)
        {
            string[] lines = new string[w.Length];
            for (int i = 0; i < w.Length; i++) lines[i] = w[i].ToString("R", CultureInfo.InvariantCulture);
            File.WriteAllLines(file, lines);
            net.loadWeights(file);
        }

        /// <summary>Activity-sheet inputs (5 inputs, 1 hidden neuron, 4 outputs): engine output vs. hand formula.</summary>
        private static bool CheckForwardPass(string file)
        {
            double[] inputs = { 1.1, 2.4, 3.2, 5.1, 3.9 };
            double[] wIn = { -0.33, 0.07, -0.45, 0.13, 0.37 };
            double[] wOut = { 0.85, 0.62, -0.10, 0.21 };
            double hiddenBias = 0.679, outBias = 0.01;

            NeuralNet net = new NeuralNet(5, 1, 4);
            List<double> p = new List<double>();
            p.AddRange(wIn); p.AddRange(wOut); p.Add(hiddenBias);
            for (int k = 0; k < 4; k++) p.Add(outBias);
            WriteParams(net, file, p.ToArray());
            for (int i = 0; i < 5; i++) net.setInputs(i, inputs[i]);
            net.run();

            double net1 = hiddenBias;
            for (int i = 0; i < 5; i++) net1 += inputs[i] * wIn[i];
            double ha = 1.0 / (1.0 + Math.Exp(-net1));
            double worst = 0;
            for (int k = 0; k < 4; k++)
            {
                double expected = 1.0 / (1.0 + Math.Exp(-(ha * wOut[k] + outBias)));
                worst = Math.Max(worst, Math.Abs(expected - net.getOuputData(k)));
            }
            bool pass = worst < 1e-12;
            Console.WriteLine("  Forward pass vs hand formula (activity inputs, HA = {0:F4}) ... {1}", ha, pass ? "ok" : "MISMATCH");
            return pass;
        }

        private static bool CheckGradients(string file)
        {
            const int nIn = 3, nHid = 4, nOut = 2;
            NeuralNet net = new NeuralNet(nIn, nHid, nOut);
            double[] x = { 0.3, 0.8, 0.5 };
            double[] t = { 0.2, 0.9 };
            for (int i = 0; i < nIn; i++) net.setInputs(i, x[i]);
            for (int o = 0; o < nOut; o++) net.setDesiredOutput(o, t[o]);

            int hiddenWeightStart = nIn * nHid;
            int hiddenBiasStart = hiddenWeightStart + nHid * nOut;
            int outputBiasStart = hiddenBiasStart + nHid;

            double[] before = ReadParams(net, file);
            net.learn();                                   // one training step
            double[] after = ReadParams(net, file);

            double[] analytic = new double[before.Length];
            for (int k = 0; k < before.Length; k++)
            {
                // input weights + hidden biases use the input-layer rate, the others the output-layer rate
                double lr = (k < hiddenWeightStart || (k >= hiddenBiasStart && k < outputBiasStart)) ? LrInput : LrOutput;
                analytic[k] = -(after[k] - before[k]) / lr;
            }
            WriteParams(net, file, before);                // restore

            double eps = 1e-6, worst = 0;
            for (int k = 0; k < before.Length; k++)
            {
                double[] w = (double[])before.Clone();
                w[k] = before[k] + eps; WriteParams(net, file, w); double up = Loss(net, t);
                w[k] = before[k] - eps; WriteParams(net, file, w); double down = Loss(net, t);
                double numeric = (up - down) / (2 * eps);
                double denom = Math.Max(1e-8, Math.Abs(numeric) + Math.Abs(analytic[k]));
                worst = Math.Max(worst, Math.Abs(numeric - analytic[k]) / denom);
            }
            bool pass = worst < 1e-5;
            Console.WriteLine("  Gradient check, {0} parameters, worst rel. error = {1:E2} ... {2}", before.Length, worst, pass ? "ok" : "MISMATCH");
            return pass;
        }

        private static double Loss(NeuralNet net, double[] t)
        {
            net.run();
            double l = 0;
            for (int o = 0; o < t.Length; o++) { double e = t[o] - net.getOuputData(o); l += 0.5 * e * e; }
            return l;
        }
    }
}