# ******************************************************************************
# import modules
# ******************************************************************************
import sys, os
import struct
import torch
import numpy as np
from os.path import join, isdir, isfile
from MathUtils import LinearTransforming
from HelperFuncs import GetWinDataPath, GetWinHomePath, GetMacOSDataPath,\
        GetDataPath, CheckTempPath

# ******************************************************************************
# MNIST class
# ******************************************************************************
class MNIST:
    def __init__(self, NumTrains=None, NumTests=None, NumEpochs=1, ScaleVal=None, \
            ScaleFlag=True, ZeroMeanFlag=False, TorchFlag=False, Verbose=False):
        """
        Python function for importing the MNIST data set.  It returns an iterator
        of 2-tuples with the first element being the label and the second element
        being a numpy.uint8 2D array of pixel data for the given image.
        """
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "MNIST::__init__()"

        # **********************************************************************
        # set the class name
        # **********************************************************************
        self.ClassName = "Mnist"

        # **********************************************************************
        # save the parameters
        # **********************************************************************
        self.NumClasses = 10
        self.NumTrains  = NumTrains
        self.NumTests   = NumTests
        self.NumEpochs  = NumEpochs
        self.ScaleVal   = ScaleVal
        self.ScaleFlag  = ScaleFlag
        self.ZeroMeanFlag = ZeroMeanFlag
        self.TorchFlag  = TorchFlag
        self.Verbose    = Verbose

        # **********************************************************************
        # reset variables
        # **********************************************************************
        self.TotalTrainingImages = None
        self.TotalTrainingLabels = None
        self.TotalNumTrains      = 60000

        self.TotalTestingImages  = None
        self.TotalTestingLabels  = None
        self.TotalNumTests       = 10000

        self.TrainingImages      = None
        self.TrainingLabels      = None
        self.TestingImages       = None
        self.TestingLabels       = None
        self.VectorLength        = None

        # **********************************************************************
        # maximum numbers of training and test data
        # **********************************************************************
        self.MaxTrains  = self.TotalNumTrains
        self.MaxTests   = self.TotalNumTests

        # **********************************************************************
        # set the variables
        # **********************************************************************
        self.Dataset    = "MNIST"
        self.SampleRate = None

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the information
            Msg = "\n==> Instantiating <%s>..." % (self.ClassName)
            print(Msg)

        # **********************************************************************
        # check the data path
        # **********************************************************************
        self.DataPath = self._CheckPath()

        # **********************************************************************
        # set training and testing data vectors
        # **********************************************************************
        self._BuildDataVectors()

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the message
            Msg = "\n==> Instantiating <%s>..." % (self.ClassName)
            print(Msg)

            # display the information
            Msg = "...%-25s: data location = %s" % (FunctionName, self.DataPath)
            print(Msg)

            # display the information
            Msg = "...%-25s: Classes = %d, Total trains = %d, total tests = %d" % \
                    (FunctionName, self.NumClasses, self.TotalNumTrains, self.TotalNumTests)
            print(Msg)

            # display the information
            Msg = "...%-25s: DataFile = %s" % (FunctionName, self.DataFile)
            print(Msg)

            # display the information
            Msg = "...%-25s: Epochs = %d, Num Trains = %d, num tests = %d, vector length = %d" % \
                    (FunctionName, self.NumEpochs, self.NumTrains, self.NumTests, self.VectorLength)
            print(Msg)

            # display the information
            Msg = "...%-25s: ScaleVal = %s, ZeroMeanFlag = %s, torch = %s" % \
                    (FunctionName, str(self.ScaleVal), str(self.ZeroMeanFlag), \
                    self.TorchFlag)
            print(Msg)

    # **************************************************************************
    # methods of the class
    # **************************************************************************
    def _CheckPath(self):
        # **********************************************************************
        # display the information
        # **********************************************************************
        FunctionName = "MNIST::_CheckPath()"

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the information
            Msg = "...%-25s: checking the data folder path ..." % (FunctionName)
            print(Msg)

        # **********************************************************************
        # Get the data path
        # **********************************************************************
        print(GetDataPath(DatasetName=self.Dataset))
        return GetDataPath(DatasetName=self.Dataset)

    # **************************************************************************
    def _LoadMNIST(self, InputType):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "MNIST::_LoadMNIST()"

        # **********************************************************************
        # check the flag
        # **********************************************************************
        if self.Verbose:
            # display the message
            Msg = "...%-25s: loading <%s> data..." % (FunctionName, InputType)
            print(Msg)
        # **********************************************************************
        # check the input type
        # **********************************************************************
        if InputType == "Training":
            ImageFileName = join(self.DataPath, "train-images.idx3-ubyte")
            LabelFileName = join(self.DataPath, "train-labels.idx1-ubyte")

        elif InputType == "Testing":
            ImageFileName = join(self.DataPath, "t10k-images.idx3-ubyte")
            LabelFileName = join(self.DataPath, "t10k-labels.idx1-ubyte")
        else:
            raise ValueError("InputType must be <Testing> or <Training>")
        
        # **********************************************************************
        # Load everything in some numpy arrays
        # **********************************************************************
        with open(LabelFileName, "rb") as LabelFd:
            Magic, Num  = struct.unpack(">II", LabelFd.read(8))
            Labels      = np.fromfile(LabelFd, dtype=np.int8)

        # **********************************************************************
        # load the image files
        # **********************************************************************
        with open(ImageFileName, "rb") as ImageFd:
            Magic, Num, Rows, Cols = struct.unpack(">IIII", ImageFd.read(16))
            Images = np.fromfile(ImageFd, dtype=np.uint8).reshape(len(Labels), Rows, Cols)
        return Images, Labels

    # **************************************************************************
    def _SetTrainLabels(self, Labels):
        # **********************************************************************
        # set the number of class
        # **********************************************************************
        NumClass = 10

        # **********************************************************************
        # get the number of labels
        # **********************************************************************
        NumLabels = len(Labels)

        # **********************************************************************
        # set the label matrix
        # **********************************************************************
        LabelMatrix = np.zeros((1, NumLabels, NumClass), dtype=float)
        for i in range(NumLabels):
            # print(i, ": Labels[i] = ", Labels[i])
            LabelMatrix[0,i,Labels[i]] = 1

        # **********************************************************************
        # check the flag
        # **********************************************************************
        if self.TorchFlag:
            return torch.from_numpy(LabelMatrix)
        else:
            return LabelMatrix

    # **************************************************************************
    def _ReadMNISTData(self):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "MNIST::_ReadMNISTData()"

        # **********************************************************************
        # check the flag
        # **********************************************************************
        if self.Verbose:
            # display the message
            Msg = "...%-25s: loading MNIST data..." % (FunctionName)
            print(Msg)

        # **********************************************************************
        # loading training images and labels
        # **********************************************************************
        self.TotalTrainingImages, self.TotalTrainingLabels = self._LoadMNIST("Training")
        self.TotalNumTrains     = len(self.TotalTrainingImages)

        # **********************************************************************
        # loading testing images and labels
        # **********************************************************************
        self.TotalTestingImages, self.TotalTestingLabels   = self._LoadMNIST("Testing")
        self.TotalNumTests      = len(self.TotalTestingImages)

    # **************************************************************************
    def _SelectSubset(self, Dataset, Labels, NumSelect):
        # set the data indices
        Indices = np.random.randint(0, len(Dataset), size=NumSelect, dtype=int)
        return Dataset[Indices], Labels[Indices]

    # **************************************************************************
    def _TrainingAndTestingSets(self):
        # set the function name
        FunctionName = "MNIST::_TrainingAndTestingSets()"

        # check the flag
        if self.Verbose:
            # display the message
            Msg = "...%-25s: loading MNIST data..." % (FunctionName)
            print(Msg)

        # check the variable before reading from the file
        if (self.TotalTrainingImages is None) or (self.TotalTestingImages is None):
            # loading from file
            self._ReadMNISTData()

        # check the number of training digits
        if (self.NumTrains is None) or (self.NumTrains == self.TotalNumTrains):
            # use the total number of training digits
            self.TrainingImages = self.TotalTrainingImages
            self.TrainingLabels = self.TotalTrainingLabels
        else:
            # set the training data set
            self.TrainingImages, self.TrainingLabels = self._SelectSubset(self.TotalTrainingImages, \
                    self.TotalTrainingLabels, self.NumTrains)

        # check the number of testing digits
        if (self.NumTests is None) or (self.NumTests == self.TotalNumTests):
            # use the total number of training digits
            self.TestingImages  = self.TotalTestingImages
            self.TestingLabels  = self.TotalTestingLabels
        else:
            # set the testing data set
            self.TestingImages, self.TestingLabels = self._SelectSubset(self.TotalTestingImages, \
                    self.TotalTestingLabels, self.NumTests)

    # **************************************************************************
    def _BuildDataVectors(self):
        # **********************************************************************
        # set the funtion name
        # **********************************************************************
        FunctionName = "MNIST::_BuildDataVectors()"

        # **********************************************************************
        # display the message
        # **********************************************************************
        if self.Verbose:
            # display the information
            Msg = "...%-25s: building MNIST training and testing data..." % \
                    (FunctionName)
            print(Msg)

        # **********************************************************************
        # check the number of trainings and testings
        # **********************************************************************
        if self.NumTrains is None:
            self.NumTrains  = self.TotalNumTrains
        if self.NumTests is None:
            self.NumTests   = self.TotalNumTests

        # **********************************************************************
        # set the file name
        # **********************************************************************
        FileName = "MNIST_Train_%d_Test_%d.npz" % (self.NumTrains, self.NumTests)

        # **********************************************************************
        # check the platform to set the data path
        # **********************************************************************
        TempPath    = CheckTempPath(FileName)
        if TempPath is not None:
            # ******************************************************************
            # update data path
            # ******************************************************************
            self.DataPath   = TempPath

        # **********************************************************************
        # set the data file
        # **********************************************************************
        self.DataFile = join(self.DataPath, FileName)

        # **********************************************************************
        # check the file status
        # **********************************************************************
        if isfile(self.DataFile):
            # ******************************************************************
            # display the information
            # ******************************************************************
            Msg = "...%-25s: loading data from file <%s>..." % \
                    (FunctionName, self.DataFile)
            print(Msg)

            # ******************************************************************
            # loading data from file
            # ******************************************************************
            Data                = np.load(self.DataFile)
            self.NumTrains      = Data["NumTrains"]
            self.TrainingImages = Data["Train"]
            self.TrainingLabels = Data["TrainLbls"]
            self.NumTests       = Data["NumTests"]
            self.TestingImages  = Data["Test"]
            self.TestingLabels  = Data["TestLbls"]

        else:
            # ******************************************************************
            # build the training and testing set
            # ******************************************************************
            self._TrainingAndTestingSets()

            # ******************************************************************
            # display the message
            # ******************************************************************
            if self.Verbose:
                # display the information
                Msg = "...%-25s: saving data to file <%s>..." % \
                        (FunctionName, self.DataFile)
                print(Msg)

            # ******************************************************************
            # save to file
            # ******************************************************************
            np.savez_compressed(self.DataFile, NumTrains=self.NumTrains,
                    Train=self.TrainingImages, TrainLbls=self.TrainingLabels,\
                    NumTests=self.NumTests, Test=self.TestingImages, \
                    TestLbls=self.TestingLabels)

        # **********************************************************************
        # set the vector length
        # **********************************************************************
        NumTrain, x, y      = self.TrainingImages.shape
        self.VectorLength   = x * y

        # **********************************************************************
        # check for torch flag
        # **********************************************************************
        if self.TorchFlag:
            self.TrainingImages = torch.from_numpy(self.TrainingImages)
            self.TrainingLabels = torch.from_numpy(self.TrainingLabels)
            self.TestingImages  = torch.from_numpy(self.TestingImages)
            self.TestingLabels  = torch.from_numpy(self.TestingLabels)

    # **************************************************************************
    def _ZeroMeanInputData(self, ScaleVal=None):
        # set the function name
        FunctionName = "MNIST::_ZeroMeanInputData()"

        # check the flag
        if self.Verbose:
            # display the message
            Msg = "...%-25s: loading zero-mean MNIST dataset..." % (FunctionName)
            print(Msg)

        # check the scale value
        if self.ScaleVal is None:
            # format error message
            Msg = "%s: ScaleVal = <%s> => invalid" % (FunctionName, str(self.ScaleVal))
            raise ValueError(Msg)

        # **********************************************************************
        # check the scale value
        # **********************************************************************
        if ScaleVal is None:
            if self.ScaleVal is None:
                # format error message
                Msg = "%s: ScaleVal = <%s> => invalid" % (FunctionName, str(self.ScaleVal))
                raise ValueError(Msg)
            else:
                AmpRange    = self.ScaleVal
        else:
            AmpRange    = ScaleVal

        # **********************************************************************
        # set the transforming y-range
        # **********************************************************************
        YRange      = [-AmpRange, AmpRange]
        BiasFlag    = True
        # print("AmpRange = ", AmpRange)

        # **********************************************************************
        # zero-mean the training images and linearly transforming data
        # **********************************************************************
        InputData   = np.subtract(self.TrainingImages, np.mean(self.TrainingImages))

        # get the image information
        NumTrains, x, y = InputData.shape

        # calculate the image size
        ImageSize = self.VectorLength

        # linearly transforming dataset
        TrainingData, Alpha, Beta = LinearTransforming(InputData, YRange, BiasFlag=BiasFlag, \
                Verbose=self.Verbose)

        # print("Before   = ", self.TrainingImages)
        # print("ZeroMean = ", InputData)
        # print("After    = ", TrainingData)

        # set the number of epochs
        NumEpochs   = self.NumEpochs

        # check the number of epoch
        if NumEpochs > 1:
            # reset the TrainingImages
            TrainingImages = np.zeros((NumEpochs, NumTrains, ImageSize))

            # copy the data
            for i in range(NumEpochs):
                TrainingImages[i] = TrainingData
        else:
            # set the training Images with epoch = 1
            TrainingImages = TrainingData.reshape(NumEpochs, NumTrains, ImageSize)

        # **********************************************************************
        # set the labels
        # **********************************************************************
        TrainingLabels = self._SetTrainLabels(self.TrainingLabels)

        # **********************************************************************
        # zero-mean the testing images and linearly transforming data
        # **********************************************************************
        InputData   = np.subtract(self.TestingImages, np.mean(self.TestingImages))

        # get the image information
        NumTests, x, y = InputData.shape

        # calculate the image size
        ImageSize = self.VectorLength

        # linearly transforming dataset
        TestingData, Alpha, Beta = LinearTransforming(InputData, YRange, BiasFlag=BiasFlag, \
                Verbose=self.Verbose)

        # set the testing Images
        TestingImages = TestingData.reshape(NumTests, ImageSize)

        # set the testing labels
        TestingLabels   = self.TestingLabels

        # set the data set
        return TrainingImages, TrainingLabels, TestingImages, TestingLabels

    # **************************************************************************
    def _RangeInputData(self, ScaleVal=None, BiasFlag=False):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "MNIST::_RangeInputData()"

        # **********************************************************************
        # check the flag
        # **********************************************************************
        if self.Verbose:
            # display the message
            Msg = "...%-25s: loading range MNIST dataset..." % (FunctionName)
            print(Msg)

        # **********************************************************************
        # check the scale value
        # **********************************************************************
        if self.ScaleVal is None:
            # format error message
            Msg = "%s: ScaleVal = <%s> => invalid" % (FunctionName, str(self.ScaleVal))
            raise ValueError(Msg)

        # **********************************************************************
        # check the scale value
        # **********************************************************************
        if ScaleVal is None:
            if self.ScaleVal is None:
                # format error message
                Msg = "%s: ScaleVal = <%s> => invalid" % (FunctionName, str(self.ScaleVal))
                raise ValueError(Msg)
            else:
                AmpRange    = self.ScaleVal
        else:
            AmpRange    = ScaleVal

        # **********************************************************************
        # set the transforming y-range
        # **********************************************************************
        YRange      = [-AmpRange, AmpRange]

        # **********************************************************************
        # get the image information
        # **********************************************************************
        NumTrains, x, y = InputData.shape

        # **********************************************************************
        # calculate the image size
        # **********************************************************************
        ImageSize = self.VectorLength

        # **********************************************************************
        # linearly transforming data
        # **********************************************************************
        TrainingData, Alpha, Beta = LinearTransforming(self.TrainingImages, YRange, BiasFlag=BiasFlag, \
                TorchFlag=self.TorchFlag, Verbose=self.Verbose)

        # **********************************************************************
        # set the number of epochs
        # **********************************************************************
        NumEpochs   = self.NumEpochs

        # **********************************************************************
        # check the number of epoch
        # **********************************************************************
        if self.TorchFlag:
            TrainingInputs  = TrainingData
        else:
            if NumEpochs > 1:
                # reset the TrainingImages
                TrainingImages = np.zeros((NumEpochs, NumTrains, ImageSize))

                # copy the data
                for i in range(NumEpochs):
                    TrainingImages[i] = TrainingData
            else:
                # set the training Images with epoch = 1
                TrainingImages = TrainingData.reshape(NumEpochs, NumTrains, ImageSize)

        # **********************************************************************
        # set the labels
        # **********************************************************************
        TrainingLabels = self._SetTrainLabels(self.TrainingLabels)

        # **********************************************************************
        # linearly transforming testing data
        # **********************************************************************
        TestingImages, Alpha, Beta = LinearTransforming(self.TestingImages, YRange, BiasFlag=BiasFlag, \
                TorchFlag=self.TorchFlag, Verbose=self.Verbose)
        TestingLabels   = self.TestingLabels

        # **********************************************************************
        # return values
        # **********************************************************************
        # print("TrainingImages   = ", TrainingImages.shape)
        # print("TrainingLabels   = ", TrainingLabels.shape)
        # print("TestingImages    = ", TestingImages.shape)
        # print("TestingLabels    = ", TestingLabels.shape)
        # exit()

        # set the data set
        return TrainingImages, TrainingLabels, TestingImages, TestingLabels

    # **************************************************************************
    def _NormalInputDataTorch(self, ScaleVal=None, TrainLblFlag=True):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "MNIST::_NormalInputDataTorch()"

        # **********************************************************************
        # check the flag
        # **********************************************************************
        if self.Verbose:
            # display the message
            Msg = "...%-25s: loading and scaling MNIST data to <%s>..." % (FunctionName, \
                    str(self.ScaleVal))
            print(Msg)

        # **********************************************************************
        # check the scale value
        # **********************************************************************
        if ScaleVal is None:
            if self.ScaleVal is None:
                # format error message
                Msg = "%s: ScaleVal = <%s> => invalid" % (FunctionName, str(self.ScaleVal))
                raise ValueError(Msg)
            else:
                AmpScale    = self.ScaleVal
        else:
            AmpScale    = ScaleVal

        # **********************************************************************
        # scaling the training images
        # **********************************************************************
        # print("AmpScale = ", AmpScale)
        Images  = self.TrainingImages
        # check the flag
        if self.ScaleFlag:
            # calculate the input scaling
            InputScaling = AmpScale/torch.max(Images)
        else:
            InputScaling = 1.0 / self.ScaleVal

        # **********************************************************************
        # set the number of epochs
        # **********************************************************************
        NumEpochs   = self.NumEpochs

        # **********************************************************************
        # get the image information
        # **********************************************************************
        NumTrains, x, y = self.TrainingImages.shape

        # **********************************************************************
        # calculate the image size
        # **********************************************************************
        ImageSize = self.VectorLength

        # **********************************************************************
        # set the training Images
        # **********************************************************************
        TrainingImages  = torch.multiply(torch.reshape(self.TrainingImages, (NumEpochs, \
                NumTrains, ImageSize)), InputScaling)

        # **********************************************************************
        # set the labels
        # **********************************************************************
        if TrainLblFlag:
            TrainingLabels  = self._SetTrainLabels(self.TrainingLabels)
        else:
            TrainingLabels  = self.TrainingLabels

        # **********************************************************************
        # get the image information
        # **********************************************************************
        NumTests, x, y = self.TestingImages.shape

        # **********************************************************************
        # calculate the image size
        # **********************************************************************
        ImageSize = self.VectorLength

        # **********************************************************************
        # set the testing Images
        # **********************************************************************
        TestingImages = torch.multiply(torch.reshape(self.TestingImages, (NumEpochs, \
                NumTests, ImageSize)), InputScaling)

        # **********************************************************************
        # set the testing labels
        # **********************************************************************
        TestingLabels   = torch.reshape(self.TestingLabels, (NumEpochs, NumTests))

        # **********************************************************************
        # return values
        # **********************************************************************
        # print("TrainingImages   = ", TrainingImages.shape)
        # print("TrainingLabels   = ", TrainingLabels.shape)
        # print("TestingImages    = ", TestingImages.shape)
        # print("TestingLabels    = ", TestingLabels.shape)
        # exit()
        return TrainingImages, TrainingLabels, TestingImages, TestingLabels

    # **************************************************************************
    def _NormalInputDataNumpy(self, ScaleVal=None, TrainLblFlag=True):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "MNIST::_NormalInputDataNumpy()"

        # **********************************************************************
        # check the flag
        # **********************************************************************
        if self.Verbose:
            # display the message
            Msg = "...%-25s: loading and scaling MNIST data to <%s>..." % (FunctionName, \
                    str(self.ScaleVal))
            print(Msg)

        # **********************************************************************
        # check the scale value
        # **********************************************************************
        if ScaleVal is None:
            if self.ScaleVal is None:
                # format error message
                Msg = "%s: ScaleVal = <%s> => invalid" % (FunctionName, str(self.ScaleVal))
                raise ValueError(Msg)
            else:
                AmpScale    = self.ScaleVal
        else:
            AmpScale    = ScaleVal

        # **********************************************************************
        # scaling the training images
        # **********************************************************************
        # print("AmpScale = ", AmpScale)
        Images  = self.TrainingImages
        # check the flag
        if self.ScaleFlag:
            # calculate the input scaling
            InputScaling = AmpScale/np.amax(Images)
        else:
            InputScaling = 1.0 / self.ScaleVal

        # **********************************************************************
        # get the image information
        # **********************************************************************
        NumTrains, x, y = Images.shape

        # **********************************************************************
        # calculate the image size
        # **********************************************************************
        ImageSize = self.VectorLength

        # **********************************************************************
        # set the training Images
        # **********************************************************************
        TrainingData = np.multiply(Images.reshape(NumTrains, ImageSize), InputScaling)

        # print("Before   = ", self.TrainingImages)
        # print("ZeroMean = ", InputData)
        # print("After    = ", TrainingData)

        # **********************************************************************
        # set the number of epochs
        # **********************************************************************
        NumEpochs   = self.NumEpochs

        # **********************************************************************
        # check the number of epoch
        # **********************************************************************
        if NumEpochs > 1:
            # reset the TrainingImages
            TrainingImages = np.zeros((NumEpochs, NumTrains, ImageSize))

            # copy the data
            for i in range(NumEpochs):
                TrainingImages[i] = TrainingData
        else:
            # set the training Images with epoch = 1
            TrainingImages = TrainingData.reshape(NumEpochs, NumTrains, ImageSize)

        # **********************************************************************
        # set the labels
        # **********************************************************************
        if TrainLblFlag:
            TrainingLabels  = self._SetTrainLabels(self.TrainingLabels)
        else:
            TrainingLabels  = self.TrainingLabels

        # **********************************************************************
        # scaling the testing images
        # **********************************************************************
        Images  = self.TestingImages

        # **********************************************************************
        # get the image information
        # **********************************************************************
        NumTests, x, y = Images.shape

        # **********************************************************************
        # calculate the image size
        # **********************************************************************
        ImageSize = self.VectorLength

        # **********************************************************************
        # set the testing Images
        # **********************************************************************
        TestingImages = np.multiply(Images.reshape(NumTests, ImageSize), InputScaling)

        # **********************************************************************
        # set the testing labels
        # **********************************************************************
        TestingLabels   = self.TestingLabels

        # **********************************************************************
        # set the data set
        # **********************************************************************
        return TrainingImages, TrainingLabels, TestingImages, TestingLabels

    # **************************************************************************
    def _NormalInputData(self, ScaleVal=None, TrainLblFlag=True):
        # **********************************************************************
        # check the flag
        # **********************************************************************
        if self.TorchFlag:
            return self._NormalInputDataTorch(ScaleVal=ScaleVal, TrainLblFlag=TrainLblFlag)
        else:
            return self._NormalInputDataNumpy(ScaleVal=ScaleVal, TrainLblFlag=TrainLblFlag)

    # **************************************************************************
    def GetDataVectors(self, ScaleVal=None, BiasFlag=False, TrainLblFlag=True):
        # **********************************************************************
        # set the function name
        # **********************************************************************
        FunctionName = "MNIST::GetDataVectors()"

        # **********************************************************************
        # check the flag
        # **********************************************************************
        if self.Verbose:
            # display the message
            Msg = "...%-25s: loading MNIST dataset, ZeroMean = <%s>..." % (FunctionName, \
                    str(self.ZeroMeanFlag))
            print(Msg)

        # **********************************************************************
        # check the scale value
        # **********************************************************************
        if ScaleVal is None:
            ScaleVal    = self.ScaleVal

        # **********************************************************************
        # check the flag:
        # **********************************************************************
        if self.ZeroMeanFlag:
            # set the dataset to a zero mean and transforming it
            return self._ZeroMeanInputData(ScaleVal=ScaleVal)
        else:
            if BiasFlag:
                return self._RangeInputData(ScaleVal=ScaleVal, BiasFlag=BiasFlag, TrainLblFlag=TrainLblFlag)
            else:
                # scaling the dataset
                return self._NormalInputData(ScaleVal=ScaleVal, TrainLblFlag=TrainLblFlag)

    # **************************************************************************
    def GetMnistInformation(self):
        return self.VectorLength, self.NumClasses

    # **************************************************************************
    def GetDatasetInformation(self):
        return self.GetMnistInformation()

    # **************************************************************************
    def GetNumEpoch(self):
        return self.NumEpochs

    # **************************************************************************
    def LoadMNISTData(self, Verbose=False):
        return [self._LoadMNIST("Training"), self._LoadMNIST("Testing")]

    # **************************************************************************
    def GetTrainingData(self):
        return self.TrainingImages, self.TrainingLabels

    # **************************************************************************
    def GetTestingData(self):
        return self.TestingImages, self.TestingLabels

    # **************************************************************************
    def GetNumTrainsAndTests(self):
        return self.NumTrains, self.NumTests

    # **************************************************************************
    def GetScaleVal(self):
        return self.ScaleVal

    # **************************************************************************
    def GetAllData(self):
        # set the function name
        FunctionName = "MNIST::GetAllData()"

        # check the flag
        if self.Verbose:
            # display the message
            Msg = "...%-25s: getting all MNIST dataset ..." % (FunctionName)
            print(Msg)

        # check the variable
        if self.TotalTrainingImages is None:
            # loading training images and labels
            self.TotalTrainingImages, self.TotalTrainingLabels = self._LoadMNIST("Training")
            self.TotalNumTrains     = len(self.TotalTrainingImages)

        # check the variable
        if self.TotalTestingImages is None:
            # loading testing images and labels
            self.TotalTestingImages, self.TotalTestingLabels   = self._LoadMNIST("Testing")
            self.TotalNumTests      = len(self.TotalTestingImages)

        # flattening training data
        NumTests, x, y  = self.TotalTrainingImages.shape
        TrainingData    = self.TotalTrainingImages.reshape(NumTests, x * y)

        # flattening testing data
        NumTests, x, y  = self.TotalTestingImages.shape
        TestingData     = self.TotalTestingImages.reshape(NumTests, x * y)

        # combining training and testing data
        AllData         = np.vstack((TrainingData, TestingData))
        AllDataLabels   = np.hstack((self.TotalTrainingLabels, self.TotalTestingLabels))
        return AllData, AllDataLabels

    # **************************************************************************
    def GetDataSetName(self):
        return self.ClassName

    # **************************************************************************
    def GetMaxTrainTest(self):
        return self.MaxTrains, self.MaxTests

    # **************************************************************************
    def ResetVerboseFlag(self):
        # reset verbose flag
        self.Verbose = False

# ******************************************************************************
# def DisplayDigit():

# ******************************************************************************
if __name__ == '__main__':
    # from MathUtils import LinearTransforming
    # import matplotlib.pyplot as plt
    # from matplotlib import rcParams
    # from pandas import DataFrame
    # rcParams.update({"figure.autolayout": True})

    # **************************************************************************
    # set the variable
    # **************************************************************************
    NumTrains   = None
    NumTests    = None
    # NumTrains   = 1000
    # NumTests    = 500
    # NumTrains   = 3000
    # NumTrains   = 6000
    # NumTests    = 1000

    Verbose     = True
    ScaleVal    = 1.0
    ScaleFlag   = True
    TorchFlag   = False
    NumEpochs   = 1
    # ZeroMeanFlag = True
    ZeroMeanFlag = False
    Verbose     = True

    # **************************************************************************
    # get the timit data set
    # **************************************************************************
    InputDataSet = MNIST(NumTrains=NumTrains, NumTests=NumTests, NumEpochs=NumEpochs, \
            ScaleVal=ScaleVal, ZeroMeanFlag=ZeroMeanFlag, TorchFlag=TorchFlag, \
            Verbose=Verbose)
    ScaleVal     = InputDataSet.GetScaleVal()
    # print("Current ScaleVal = ", ScaleVal)

    # **************************************************************************
    # get input images
    # **************************************************************************
    TrainInputs, TrainLbls, TestInputs, TestLbls = InputDataSet.GetDataVectors(ScaleVal=1.0)

    print("Train input images   = ", TrainInputs.shape)
    print("Train input labels   = ", TrainLbls.shape)
    print("Test input images    = ", TestInputs.shape)
    print("Test input labels    = ", TestLbls.shape)

    exit()

    # # get the training and testing dataset
    # # TrainImgs, TrainLbls, TestImgs, TestLbls = MNISTData.GetDataVectors(ScaleVal=ScaleVal, \
    # #         ScaleFlag=ScaleFlag, Verbose=Verbose)
    # TrainImgs, TrainLbls, TestImgs, TestLbls = MNISTData.GetDataVectors(ScaleVal=ScaleVal, \
    #         ScaleFlag=ScaleFlag, ZeroMeanFlag=ZeroMeanFlag, Verbose=Verbose)
    #
    # print("TrainImgs   = ", TrainImgs.shape)
    # print("TrainLbls   = ", TrainLbls.shape)
    # print("TestImgs    = ", TestImgs.shape)
    # print("TestLbls    = ", TestLbls.shape)

    AllData, AllDataLabels = MNISTData.GetAllData()
    print("AllData        = ", AllData.shape)
    print("AllDataLabels  = ", AllDataLabels.shape)

    exit()

    # set the line width
    LineWidth = 1.5
    # LineWidth = 2.0
    # LineWidth = 2.5

    # set the font family
    FontSize = 12
    # FontSize = 14
    # FontSize = 16
    # FontSize = 18
    # FontSize = 20
    # FontSize = 22
    # FontSize = 24
    # FontSize = 28
    font = {"family": "Times New Roman", "size": FontSize}
    plt.rc("font", **font)  # pass in the font dict as kwargs

    # # check platform
    # if (Platform == "linux2") or (Platform == "linux"):
    #     # set latex
    #     plt.rc("text", usetex=True)

    # set the title
    # FigureTitle = "MFCC frame %d" % (FramNum)
    # Fig = plt.figure(FigureTitle)
    # plt.grid(linestyle="dotted")
    # # plt.plot(t, RhoSpice, "-", label="RhoSpice", linewidth=LineWidth)
    # plt.plot(ResultMfcc[FramNum], "-", label="MFCC", linewidth=LineWidth)
    # plt.plot(Test, "--", label="MFCC[1]_Mean", linewidth=LineWidth)
    # plt.xlabel("Index")
    # plt.ylabel("MFCCs")
    # plt.axis("tight")
    # plt.legend(loc="best")

    # plt.show()
