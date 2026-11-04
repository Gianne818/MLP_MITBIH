using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace MLP_MITBIH
{
  public class DataPreprocessor
  {
    public class RawSample
    {
      public double[] Features { get; set; } = null!;
      public int Label { get; set; }
    }

    // reads and validates CSV rows, skipping headers and dropping malformed/NaN lines
    public static List<RawSample> LoadCsv(string csvPath, out int numFeatures, out int numClasses)
    {
      if (!File.Exists(csvPath))
        throw new FileNotFoundException("Dataset CSV not found at: " + csvPath);

      List<RawSample> samples = new List<RawSample>();
      string[] lines = File.ReadAllLines(csvPath);

      if (lines.Length <= 1)
        throw new Exception("Dataset file is empty or only contains a header.");

      // detect header by checking whether the first column is numeric
      int startLine = 0;
      string[] firstTokens = lines[0].Split(',');
      if (!double.TryParse(firstTokens[0], NumberStyles.Any, CultureInfo.InvariantCulture, out _))
      {
        startLine = 1;
      }

      numFeatures = firstTokens.Length - 1;
      HashSet<int> uniqueLabels = new HashSet<int>();

      for (int i = startLine; i < lines.Length; ++i)
      {
        string line = lines[i].Trim();
        if (string.IsNullOrWhiteSpace(line)) continue;

        string[] tokens = line.Split(',');
        if (tokens.Length != numFeatures + 1) continue;

        double[] feat = new double[numFeatures];
        bool validRow = true;

        for (int c = 0; c < numFeatures; ++c)
        {
          if (double.TryParse(tokens[c], NumberStyles.Any, CultureInfo.InvariantCulture, out double val))
          {
            if (double.IsNaN(val) || double.IsInfinity(val))
            {
              validRow = false;
              break;
            }
            feat[c] = val;
          }
          else
          {
            validRow = false;
            break;
          }
        }

        // support integer or floating-point label formatting (e.g., 0 or 0.0)
        if (!int.TryParse(tokens[numFeatures], NumberStyles.Any, CultureInfo.InvariantCulture, out int label))
        {
          if (double.TryParse(tokens[numFeatures], NumberStyles.Any, CultureInfo.InvariantCulture, out double labelDouble))
            label = (int)labelDouble;
          else
            validRow = false;
        }

        if (validRow)
        {
          samples.Add(new RawSample { Features = feat, Label = label });
          uniqueLabels.Add(label);
        }
      }

      numClasses = uniqueLabels.Count;
      Console.WriteLine($"[DataPreprocessor] Loaded {samples.Count} valid samples with {numFeatures} features and {numClasses} distinct classes.");
      return samples;
    }

    // stratified split preserves identical class proportions across train and test sets
    public static void StratifiedSplit(List<RawSample> allSamples, double trainPct, int seed,
      out List<RawSample> trainList, out List<RawSample> testList)
    {
      Random rnd = new Random(seed);

      var grouped = new Dictionary<int, List<RawSample>>();
      foreach (var s in allSamples)
      {
        if (!grouped.ContainsKey(s.Label))
          grouped[s.Label] = new List<RawSample>();
        grouped[s.Label].Add(s);
      }

      trainList = new List<RawSample>();
      testList = new List<RawSample>();

      foreach (var kvp in grouped)
      {
        var classList = kvp.Value;

        // shuffle within each class partition before splitting
        for (int i = 0; i < classList.Count; ++i)
        {
          int r = rnd.Next(i, classList.Count);
          var temp = classList[r];
          classList[r] = classList[i];
          classList[i] = temp;
        }

        int trainCount = (int)(classList.Count * trainPct);
        for (int i = 0; i < trainCount; ++i)
          trainList.Add(classList[i]);
        for (int i = trainCount; i < classList.Count; ++i)
          testList.Add(classList[i]);
      }

      // mix all classes in training set so mini-batches/iterations don't see grouped classes
      for (int i = 0; i < trainList.Count; ++i)
      {
        int r = rnd.Next(i, trainList.Count);
        var temp = trainList[r];
        trainList[r] = trainList[i];
        trainList[i] = temp;
      }
    }

    public class MinMaxModel
    {
      public double[] MinValues { get; set; } = null!;
      public double[] MaxValues { get; set; } = null!;
    }

    // fit min/max strictly on training data to avoid data leakage
    public static MinMaxModel FitMinMax(List<RawSample> trainSamples, int numFeatures)
    {
      double[] minVals = new double[numFeatures];
      double[] maxVals = new double[numFeatures];

      for (int j = 0; j < numFeatures; ++j)
      {
        minVals[j] = trainSamples[0].Features[j];
        maxVals[j] = trainSamples[0].Features[j];
      }

      for (int i = 1; i < trainSamples.Count; ++i)
      {
        double[] f = trainSamples[i].Features;
        for (int j = 0; j < numFeatures; ++j)
        {
          if (f[j] < minVals[j]) minVals[j] = f[j];
          if (f[j] > maxVals[j]) maxVals[j] = f[j];
        }
      }

      return new MinMaxModel { MinValues = minVals, MaxValues = maxVals };
    }

    // scales features to [0, 1] using (x - min) / (max - min) and appends 1-of-N one-hot target
    public static double[][] TransformAndOneHot(List<RawSample> samples, MinMaxModel scaler, int numFeatures, int numClasses)
    {
      double[][] matrix = new double[samples.Count][];

      for (int i = 0; i < samples.Count; ++i)
      {
        matrix[i] = new double[numFeatures + numClasses];

        for (int j = 0; j < numFeatures; ++j)
        {
          double min = scaler.MinValues[j];
          double max = scaler.MaxValues[j];
          double range = max - min;

          if (Math.Abs(range) < 1.0e-9)
          {
            matrix[i][j] = 0.5;
          }
          else
          {
            double scaled = (samples[i].Features[j] - min) / range;
            if (scaled < 0.0) scaled = 0.0;
            if (scaled > 1.0) scaled = 1.0;
            matrix[i][j] = scaled;
          }
        }

        // one-hot encode the target label into the last numClasses columns
        int label = samples[i].Label;
        for (int k = 0; k < numClasses; ++k)
        {
          matrix[i][numFeatures + k] = (k == label) ? 1.0 : 0.0;
        }
      }

      return matrix;
    }
  }
}
