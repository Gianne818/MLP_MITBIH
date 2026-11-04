using System;
using System.Collections.Generic;

namespace MLP_MITBIH
{
  // 3-layer Multi-Layer Perceptron (Input -> Hidden -> Output) with backpropagation
  public class NeuralNetwork
  {
    private int numInput;
    private int numHidden;
    private int numOutput;

    private double[] inputs;
    private double[][] ihWeights; // input-to-hidden weights
    private double[] hBiases;     // hidden layer biases
    private double[] hOutputs;    // hidden layer activated outputs

    private double[][] hoWeights; // hidden-to-output weights
    private double[] oBiases;     // output layer biases
    private double[] outputs;     // softmax normalized output probabilities

    private Random rnd;

    public NeuralNetwork(int numInput, int numHidden, int numOutput, int seed = 0)
    {
      this.numInput = numInput;
      this.numHidden = numHidden;
      this.numOutput = numOutput;

      this.inputs = new double[numInput];

      this.ihWeights = MakeMatrix(numInput, numHidden, 0.0);
      this.hBiases = new double[numHidden];
      this.hOutputs = new double[numHidden];

      this.hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
      this.oBiases = new double[numOutput];
      this.outputs = new double[numOutput];

      this.rnd = new Random(seed);
      this.InitializeWeights();
    }

    private static double[][] MakeMatrix(int rows, int cols, double v)
    {
      double[][] result = new double[rows][];
      for (int r = 0; r < result.Length; ++r)
        result[r] = new double[cols];

      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
          result[i][j] = v;

      return result;
    }

    // symmetric Xavier-scale weight initialization to keep tanh in its active linear range
    private void InitializeWeights()
    {
      int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
      double[] initialWeights = new double[numWeights];
      double scale = Math.Sqrt(2.0 / (numInput + numHidden));

      for (int i = 0; i < initialWeights.Length; ++i)
        initialWeights[i] = (2.0 * rnd.NextDouble() - 1.0) * scale;

      this.SetWeights(initialWeights);
    }

    public void SetWeights(double[] weights)
    {
      int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
      if (weights.Length != numWeights)
        throw new Exception("Bad weights array length in SetWeights: expected " + numWeights + ", got " + weights.Length);

      int k = 0;

      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
          ihWeights[i][j] = weights[k++];

      for (int i = 0; i < numHidden; ++i)
        hBiases[i] = weights[k++];

      for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
          hoWeights[i][j] = weights[k++];

      for (int i = 0; i < numOutput; ++i)
        oBiases[i] = weights[k++];
    }

    public double[] GetWeights()
    {
      int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
      double[] result = new double[numWeights];
      int k = 0;

      for (int i = 0; i < ihWeights.Length; ++i)
        for (int j = 0; j < ihWeights[0].Length; ++j)
          result[k++] = ihWeights[i][j];

      for (int i = 0; i < hBiases.Length; ++i)
        result[k++] = hBiases[i];

      for (int i = 0; i < hoWeights.Length; ++i)
        for (int j = 0; j < hoWeights[0].Length; ++j)
          result[k++] = hoWeights[i][j];

      for (int i = 0; i < oBiases.Length; ++i)
        result[k++] = oBiases[i];

      return result;
    }

    // forward pass: input features -> tanh hidden -> softmax output probabilities
    public double[] ComputeOutputs(double[] xValues)
    {
      double[] hSums = new double[numHidden];
      double[] oSums = new double[numOutput];

      for (int i = 0; i < xValues.Length; ++i)
        this.inputs[i] = xValues[i];

      // hidden layer linear sum
      for (int j = 0; j < numHidden; ++j)
        for (int i = 0; i < numInput; ++i)
          hSums[j] += this.inputs[i] * this.ihWeights[i][j];

      for (int i = 0; i < numHidden; ++i)
        hSums[i] += this.hBiases[i];

      // hidden activation: tanh
      for (int i = 0; i < numHidden; ++i)
        this.hOutputs[i] = HyperTan(hSums[i]);

      // output layer linear sum
      for (int j = 0; j < numOutput; ++j)
        for (int i = 0; i < numHidden; ++i)
          oSums[j] += this.hOutputs[i] * this.hoWeights[i][j];

      for (int i = 0; i < numOutput; ++i)
        oSums[i] += this.oBiases[i];

      // output activation: softmax
      double[] softOut = Softmax(oSums);
      Array.Copy(softOut, this.outputs, softOut.Length);

      double[] retResult = new double[numOutput];
      Array.Copy(this.outputs, retResult, retResult.Length);
      return retResult;
    }

    private static double HyperTan(double x)
    {
      if (x < -20.0) return -1.0;
      else if (x > 20.0) return 1.0;
      else return Math.Tanh(x);
    }

    // softmax with max-shift subtraction to prevent exp() overflow (NaN)
    private static double[] Softmax(double[] oSums)
    {
      double max = oSums[0];
      for (int i = 1; i < oSums.Length; ++i)
        if (oSums[i] > max)
          max = oSums[i];

      double sum = 0.0;
      for (int i = 0; i < oSums.Length; ++i)
        sum += Math.Exp(oSums[i] - max);

      double[] result = new double[oSums.Length];
      for (int i = 0; i < oSums.Length; ++i)
        result[i] = Math.Exp(oSums[i] - max) / sum;

      return result;
    }

    public class EpochLog
    {
      public int Epoch { get; set; }
      public double TrainLoss { get; set; }
      public double TestLoss { get; set; }
      public double TrainAccuracy { get; set; }
      public double TestAccuracy { get; set; }
    }

    // trains using backpropagation with momentum and L2 weight decay
    public List<EpochLog> Train(double[][] trainData, double[][]? testData, int maxEpochs, double learnRate, double momentum, double decay, int logInterval = 10, bool verbose = true)
    {
      double[][] hoGrads = MakeMatrix(numHidden, numOutput, 0.0);
      double[] obGrads = new double[numOutput];

      double[][] ihGrads = MakeMatrix(numInput, numHidden, 0.0);
      double[] hbGrads = new double[numHidden];

      double[] oSignals = new double[numOutput];
      double[] hSignals = new double[numHidden];

      // momentum velocity buffers
      double[][] ihPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
      double[] hPrevBiasesDelta = new double[numHidden];
      double[][] hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
      double[] oPrevBiasesDelta = new double[numOutput];

      int epoch = 0;
      double[] xValues = new double[numInput];
      double[] tValues = new double[numOutput];
      double derivative = 0.0;

      int[] sequence = new int[trainData.Length];
      for (int i = 0; i < sequence.Length; ++i)
        sequence[i] = i;

      List<EpochLog> history = new List<EpochLog>();

      while (epoch < maxEpochs)
      {
        ++epoch;

        if (epoch % logInterval == 0 || epoch == 1 || epoch == maxEpochs)
        {
          double trainErr = Error(trainData);
          double trainAcc = Accuracy(trainData);
          double testErr = testData != null ? Error(testData) : 0.0;
          double testAcc = testData != null ? Accuracy(testData) : 0.0;

          history.Add(new EpochLog
          {
            Epoch = epoch,
            TrainLoss = trainErr,
            TestLoss = testErr,
            TrainAccuracy = trainAcc,
            TestAccuracy = testAcc
          });

          if (verbose)
          {
            Console.WriteLine($"Epoch {epoch,4}/{maxEpochs} | Train Loss: {trainErr:F4} | Train Acc: {trainAcc * 100,6:F2}% | Test Loss: {testErr:F4} | Test Acc: {testAcc * 100,6:F2}%");
          }
        }

        Shuffle(sequence);

        for (int ii = 0; ii < trainData.Length; ++ii)
        {
          int idx = sequence[ii];

          Array.Copy(trainData[idx], 0, xValues, 0, numInput);
          Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);

          ComputeOutputs(xValues);

          // 1. Output error signals: (target - output) for softmax + cross-entropy
          for (int k = 0; k < numOutput; ++k)
            oSignals[k] = tValues[k] - outputs[k];

          // 2. Hidden-to-output gradients
          for (int j = 0; j < numHidden; ++j)
            for (int k = 0; k < numOutput; ++k)
              hoGrads[j][k] = oSignals[k] * hOutputs[j];

          for (int k = 0; k < numOutput; ++k)
            obGrads[k] = oSignals[k] * 1.0;

          // 3. Backpropagate error signals through tanh derivative: (1 + h) * (1 - h)
          for (int j = 0; j < numHidden; ++j)
          {
            derivative = (1.0 + hOutputs[j]) * (1.0 - hOutputs[j]);
            double sum = 0.0;
            for (int k = 0; k < numOutput; ++k)
              sum += oSignals[k] * hoWeights[j][k];
            hSignals[j] = derivative * sum;
          }

          // 4. Input-to-hidden gradients
          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
              ihGrads[i][j] = hSignals[j] * inputs[i];

          for (int j = 0; j < numHidden; ++j)
            hbGrads[j] = hSignals[j] * 1.0;

          // update input-to-hidden weights with learning rate, decay, and momentum
          for (int i = 0; i < numInput; ++i)
          {
            for (int j = 0; j < numHidden; ++j)
            {
              double delta = ihGrads[i][j] * learnRate;
              ihWeights[i][j] += delta;
              ihWeights[i][j] -= (learnRate * decay * ihWeights[i][j]);
              ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
              ihPrevWeightsDelta[i][j] = delta;
            }
          }

          // update hidden biases
          for (int j = 0; j < numHidden; ++j)
          {
            double delta = hbGrads[j] * learnRate;
            hBiases[j] += delta;
            hBiases[j] += hPrevBiasesDelta[j] * momentum;
            hPrevBiasesDelta[j] = delta;
          }

          // update hidden-to-output weights with learning rate, decay, and momentum
          for (int j = 0; j < numHidden; ++j)
          {
            for (int k = 0; k < numOutput; ++k)
            {
              double delta = hoGrads[j][k] * learnRate;
              hoWeights[j][k] += delta;
              hoWeights[j][k] -= (learnRate * decay * hoWeights[j][k]);
              hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
              hoPrevWeightsDelta[j][k] = delta;
            }
          }

          // update output biases
          for (int k = 0; k < numOutput; ++k)
          {
            double delta = obGrads[k] * learnRate;
            oBiases[k] += delta;
            oBiases[k] += oPrevBiasesDelta[k] * momentum;
            oPrevBiasesDelta[k] = delta;
          }
        }
      }

      return history;
    }

    private void Shuffle(int[] sequence)
    {
      for (int i = 0; i < sequence.Length; ++i)
      {
        int r = this.rnd.Next(i, sequence.Length);
        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }
    }

    // categorical cross-entropy loss with probability clamping to prevent log(0)
    public double Error(double[][] data)
    {
      double sumCrossEntropy = 0.0;
      double[] xValues = new double[numInput];
      double[] tValues = new double[numOutput];

      for (int i = 0; i < data.Length; ++i)
      {
        Array.Copy(data[i], 0, xValues, 0, numInput);
        Array.Copy(data[i], numInput, tValues, 0, numOutput);
        double[] yValues = this.ComputeOutputs(xValues);

        for (int j = 0; j < numOutput; ++j)
        {
          if (tValues[j] == 1.0)
          {
            double prob = Math.Max(yValues[j], 1.0e-15);
            sumCrossEntropy += Math.Log(prob);
          }
        }
      }
      return -sumCrossEntropy / data.Length;
    }

    // overall classification accuracy using argmax winner-takes-all
    public double Accuracy(double[][] data)
    {
      int numCorrect = 0;
      double[] xValues = new double[numInput];
      double[] tValues = new double[numOutput];

      for (int i = 0; i < data.Length; ++i)
      {
        Array.Copy(data[i], 0, xValues, 0, numInput);
        Array.Copy(data[i], numInput, tValues, 0, numOutput);
        double[] yValues = this.ComputeOutputs(xValues);

        int predIndex = MaxIndex(yValues);
        int trueIndex = MaxIndex(tValues);

        if (predIndex == trueIndex)
          ++numCorrect;
      }
      return (numCorrect * 1.0) / data.Length;
    }

    public static int MaxIndex(double[] vector)
    {
      int bigIndex = 0;
      double biggestVal = vector[0];
      for (int i = 1; i < vector.Length; ++i)
      {
        if (vector[i] > biggestVal)
        {
          biggestVal = vector[i];
          bigIndex = i;
        }
      }
      return bigIndex;
    }
  }
}
