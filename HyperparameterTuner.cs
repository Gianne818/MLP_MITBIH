using System;
using System.Collections.Generic;
using System.IO;

namespace MLP_MITBIH
{
  // systematic grid search over hidden layer sizes, learning rates, and epochs
  public class HyperparameterTuner
  {
    public class TrialResult
    {
      public int TrialId { get; set; }
      public int NumHidden { get; set; }
      public double LearnRate { get; set; }
      public int MaxEpochs { get; set; }
      public double Momentum { get; set; }
      public double Decay { get; set; }
      public double TrainLoss { get; set; }
      public double TestLoss { get; set; }
      public double TrainAccuracy { get; set; }
      public double TestAccuracy { get; set; }
      public double MacroF1 { get; set; }
    }

    public static List<TrialResult> RunGridSearch(
      double[][] trainData,
      double[][] testData,
      int numInput,
      int numOutput,
      int[] hiddenCandidates,
      double[] lrCandidates,
      int[] epochCandidates,
      double momentum = 0.01,
      double decay = 0.0000005,
      string csvExportPath = "outputs/tuning_results.csv")
    {
      Console.WriteLine("================================================================================");
      Console.WriteLine(" STARTING HYPERPARAMETER TUNING EXPERIMENTS");
      Console.WriteLine("================================================================================");
      Console.WriteLine($"Candidate Hidden Layer Sizes : [{string.Join(", ", hiddenCandidates)}]");
      Console.WriteLine($"Candidate Learning Rates     : [{string.Join(", ", lrCandidates)}]");
      Console.WriteLine($"Candidate Epoch Sizes        : [{string.Join(", ", epochCandidates)}]");
      Console.WriteLine("--------------------------------------------------------------------------------");

      List<TrialResult> trials = new List<TrialResult>();
      int trialId = 1;
      int totalTrials = hiddenCandidates.Length * lrCandidates.Length * epochCandidates.Length;

      foreach (int hidden in hiddenCandidates)
      {
        foreach (double lr in lrCandidates)
        {
          foreach (int epochs in epochCandidates)
          {
            Console.Write($"[Trial {trialId,2}/{totalTrials}] Hidden={hidden,2}, LR={lr:F3}, Epochs={epochs,3} ... ");

            NeuralNetwork nn = new NeuralNetwork(numInput, hidden, numOutput, seed: 42);
            nn.Train(trainData, testData, epochs, lr, momentum, decay, logInterval: epochs, verbose: false);

            EvaluationMetrics.Report rep = EvaluationMetrics.Evaluate(nn, testData, numInput, numOutput);
            double trainLoss = nn.Error(trainData);
            double trainAcc = nn.Accuracy(trainData);

            TrialResult tr = new TrialResult
            {
              TrialId = trialId++,
              NumHidden = hidden,
              LearnRate = lr,
              MaxEpochs = epochs,
              Momentum = momentum,
              Decay = decay,
              TrainLoss = trainLoss,
              TestLoss = rep.Loss,
              TrainAccuracy = trainAcc,
              TestAccuracy = rep.Accuracy,
              MacroF1 = rep.MacroF1
            };

            trials.Add(tr);
            Console.WriteLine($"Test Loss: {tr.TestLoss:F4} | Test Acc: {tr.TestAccuracy * 100:F2}% | Macro-F1: {tr.MacroF1 * 100:F2}%");
          }
        }
      }

      ExportResultsToCsv(trials, csvExportPath);
      PrintSummary(trials);

      return trials;
    }

    private static void ExportResultsToCsv(List<TrialResult> trials, string path)
    {
      string? dir = Path.GetDirectoryName(path);
      if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        Directory.CreateDirectory(dir);

      using (StreamWriter sw = new StreamWriter(path))
      {
        sw.WriteLine("TrialId,NumHidden,LearnRate,MaxEpochs,Momentum,Decay,TrainLoss,TestLoss,TrainAccuracy,TestAccuracy,MacroF1");
        foreach (var t in trials)
        {
          sw.WriteLine($"{t.TrialId},{t.NumHidden},{t.LearnRate:F4},{t.MaxEpochs},{t.Momentum},{t.Decay},{t.TrainLoss:F6},{t.TestLoss:F6},{t.TrainAccuracy:F6},{t.TestAccuracy:F6},{t.MacroF1:F6}");
        }
      }
      Console.WriteLine($"\n[HyperparameterTuner] Saved {trials.Count} experiment logs to {path}");
    }

    private static void PrintSummary(List<TrialResult> trials)
    {
      Console.WriteLine("\n================================================================================");
      Console.WriteLine(" HYPERPARAMETER TUNING LEADERBOARD (Ranked by Macro-F1 Score)");
      Console.WriteLine("================================================================================");
      Console.WriteLine($"{"Rank",-5} {"Trial",-6} {"Hidden",-7} {"LearnRate",-10} {"Epochs",-8} {"Test Loss",-10} {"Test Acc",-10} {"Macro-F1",-10}");
      Console.WriteLine(new string('-', 72));

      var sorted = new List<TrialResult>(trials);
      sorted.Sort((a, b) => b.MacroF1.CompareTo(a.MacroF1));

      for (int i = 0; i < sorted.Count; ++i)
      {
        var t = sorted[i];
        string star = (i == 0) ? " (Optimal)" : "";
        Console.WriteLine($"{i + 1,-5} {t.TrialId,-6} {t.NumHidden,-7} {t.LearnRate,-10:F3} {t.MaxEpochs,-8} {t.TestLoss,-10:F4} {t.TestAccuracy * 100,8:F2}% {t.MacroF1 * 100,8:F2}%{star}");
      }
      Console.WriteLine("================================================================================\n");
    }
  }
}
