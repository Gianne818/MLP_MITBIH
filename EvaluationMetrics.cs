using System;

namespace MLP_MITBIH
{
  // computes confusion matrix, accuracy, per-class Precision/Recall/F1, and macro/weighted averages
  public class EvaluationMetrics
  {
    public static readonly string[] ClassNames = new string[]
    {
      "N (Normal)",
      "S (Supraventricular)",
      "V (Ventricular)",
      "F (Fusion)",
      "Q (Unknown/Paced)"
    };

    public class Report
    {
      public int TotalSamples { get; set; }
      public int NumClasses { get; set; }
      public double Accuracy { get; set; }
      public double Loss { get; set; }
      public int[][] ConfusionMatrix { get; set; } = null!;
      public double[] Precision { get; set; } = null!;
      public double[] Recall { get; set; } = null!;
      public double[] F1 { get; set; } = null!;
      public int[] Support { get; set; } = null!;
      public double MacroPrecision { get; set; }
      public double MacroRecall { get; set; }
      public double MacroF1 { get; set; }
      public double WeightedF1 { get; set; }
    }

    public static Report Evaluate(NeuralNetwork nn, double[][] data, int numInput, int numOutput)
    {
      int n = data.Length;
      int[][] cm = new int[numOutput][];
      for (int i = 0; i < numOutput; ++i)
        cm[i] = new int[numOutput];

      double[] xValues = new double[numInput];
      double[] tValues = new double[numOutput];
      int correct = 0;

      for (int i = 0; i < n; ++i)
      {
        Array.Copy(data[i], 0, xValues, 0, numInput);
        Array.Copy(data[i], numInput, tValues, 0, numOutput);

        double[] yValues = nn.ComputeOutputs(xValues);
        int pred = NeuralNetwork.MaxIndex(yValues);
        int actual = NeuralNetwork.MaxIndex(tValues);

        cm[actual][pred]++;
        if (pred == actual)
          correct++;
      }

      double[] precision = new double[numOutput];
      double[] recall = new double[numOutput];
      double[] f1 = new double[numOutput];
      int[] support = new int[numOutput];

      double sumPrec = 0.0;
      double sumRec = 0.0;
      double sumF1 = 0.0;
      double weightedF1Sum = 0.0;

      for (int c = 0; c < numOutput; ++c)
      {
        int tp = cm[c][c];

        // false positives: column sum minus diagonal
        int fp = 0;
        for (int r = 0; r < numOutput; ++r)
          if (r != c) fp += cm[r][c];

        // false negatives: row sum minus diagonal
        int fn = 0;
        for (int col = 0; col < numOutput; ++col)
          if (col != c) fn += cm[c][col];

        support[c] = tp + fn;
        precision[c] = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
        recall[c] = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
        f1[c] = (precision[c] + recall[c] > 0)
          ? 2.0 * (precision[c] * recall[c]) / (precision[c] + recall[c])
          : 0.0;

        sumPrec += precision[c];
        sumRec += recall[c];
        sumF1 += f1[c];
        weightedF1Sum += f1[c] * support[c];
      }

      double loss = nn.Error(data);

      return new Report
      {
        TotalSamples = n,
        NumClasses = numOutput,
        Accuracy = (double)correct / n,
        Loss = loss,
        ConfusionMatrix = cm,
        Precision = precision,
        Recall = recall,
        F1 = f1,
        Support = support,
        MacroPrecision = sumPrec / numOutput,
        MacroRecall = sumRec / numOutput,
        MacroF1 = sumF1 / numOutput,
        WeightedF1 = (n > 0) ? weightedF1Sum / n : 0.0
      };
    }

    public static void PrintReport(Report r, string title)
    {
      Console.WriteLine("================================================================================");
      Console.WriteLine($" {title}");
      Console.WriteLine("================================================================================");
      Console.WriteLine($"Total Samples: {r.TotalSamples} | Loss: {r.Loss:F4} | Overall Accuracy: {r.Accuracy * 100:F2}%\n");

      // confusion Matrix display
      Console.WriteLine("Confusion Matrix (Rows: Actual, Columns: Predicted):");
      Console.Write("Actual \\ Pred | ");
      for (int c = 0; c < r.NumClasses; ++c)
      {
        string header = (c < ClassNames.Length) ? ClassNames[c].Split(' ')[0] : $"C{c}";
        Console.Write($"{header,8} ");
      }
      Console.WriteLine("| Total");
      Console.WriteLine(new string('-', 68));

      for (int i = 0; i < r.NumClasses; ++i)
      {
        string rowName = (i < ClassNames.Length) ? ClassNames[i].Split(' ')[0] : $"C{i}";
        Console.Write($"{rowName,-13} | ");
        for (int j = 0; j < r.NumClasses; ++j)
        {
          Console.Write($"{r.ConfusionMatrix[i][j],8} ");
        }
        Console.WriteLine($"| {r.Support[i],6}");
      }
      Console.WriteLine(new string('-', 68));

      // detailed classification metrics table
      Console.WriteLine("\nClassification Report:");
      Console.WriteLine($"{"Class",-24} {"Precision",10} {"Recall",10} {"F1-Score",10} {"Support",10}");
      Console.WriteLine(new string('-', 68));

      for (int c = 0; c < r.NumClasses; ++c)
      {
        string name = (c < ClassNames.Length) ? ClassNames[c] : $"Class {c}";
        Console.WriteLine($"{name,-24} {r.Precision[c] * 100,9:F2}% {r.Recall[c] * 100,9:F2}% {r.F1[c] * 100,9:F2}% {r.Support[c],10}");
      }

      Console.WriteLine(new string('-', 68));
      Console.WriteLine($"{"Macro Average",-24} {r.MacroPrecision * 100,9:F2}% {r.MacroRecall * 100,9:F2}% {r.MacroF1 * 100,9:F2}% {r.TotalSamples,10}");
      Console.WriteLine($"{"Weighted Average",-24} {"-",10} {"-",10} {r.WeightedF1 * 100,9:F2}% {r.TotalSamples,10}");
      Console.WriteLine("================================================================================\n");
    }
  }
}
