SR = 44100

FFT_SIZE = 4096
H = int(FFT_SIZE/4)
BATCH_SIZE = 6

PATCH_SIZE = 64

lr = 0.0005

WINDOW = "hanning"

PATH_FFT = "DataSet/fft/"
PATH_PHASE = "DataSet/fft_phase"
PATH_VAL_DATA = "Result"
PATH_CHECK_DATA = "Audiocheck"
TESTDATA_PATH = "TestData"
PATH_MODEL = "./checkpoint/"
