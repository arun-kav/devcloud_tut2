import os
import logging as log

from PIL import Image
import numpy as np
import cv2
import sys
import os
from argparse import ArgumentParser
from qarpo.demoutils import *
import applicationMetricWriter
from time import time

#SETUP THE DATASET
import _pickle as cPickle
import tensorflow as tf

#make sure this file is in your current directory!
with open("RML2016.10a_dict.pkl", 'rb') as f:
    Xd = cPickle.load(f, encoding="latin1") 

#setup the input data
#snr = signal to noise ratio
# mod is the label. mod = modulation scheme
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])

#Let X be the array of modulations
X = []
labels = []

for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  labels.append((mod,snr))
X = np.vstack(X)

#Setup inference engine
try:
    from openvino import inference_engine as ie
    from openvino.inference_engine import IENetwork, IECore, IEPlugin
    
except Exception as e:
    exception_type = type(e).__name__
    print("The following error happened while importing Python API module:\n[ {} ] {}".format(exception_type, e))
    sys.exit(1)

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("-ni", "--number_iter", help="Number of inference iterations", default=1, type=int)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("--num_threads", default=88, type=int)
    parser.add_argument("-nt", "--number_top", help="Number of top results", default=11, type=int)
    parser.add_argument("-p", "--out_dir", help="Optional. The path where result files and logs will be stored",
                      required=False, default="./results", type=str)
    parser.add_argument("-o", "--out_prefix", 
                      help="Optional. The file name prefix in the output_directory where results will be stored", 
                      required=False, default="out_", type=str)
    parser.add_argument("-g", "--log_prefix", 
                      help="Optional. The file name prefix in the output directory for log files",
                      required=False, default="log_", type=str)

    return parser

def main():
    # Run inference
    job_id = os.getenv("PBS_JOBID")    
    
    # Plugin initialization for specified device and load extensions library if specified.
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Set up logging to a file
    logpath = os.path.join(os.path.dirname(__file__), ".log")
    log.basicConfig(level=log.INFO,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                    filename=logpath,
                    filemode="w" )
    try:
        job_id = os.environ['PBS_JOBID']
        infer_file = os.path.join(args.out_dir,'i_progress_'+str(job_id)+'.txt')
    except Exception as e:
        log.warning(e)
        log.warning("job_id: {}".format(job_id))
    
    # Setup additional logging to console
    console = log.StreamHandler()
    console.setLevel(log.INFO)
    formatter = log.Formatter("[ %(levelname)s ] %(message)s")
    console.setFormatter(formatter)
    log.getLogger("").addHandler(console)
    
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    ie = IECore()
   
    if args.cpu_extension and 'CPU' in args.device:
        log.info("Loading plugins for {} device...".format(args.device))
        ie.add_extension(args.cpu_extension, "CPU")

    # Read IR
    log.info("Reading IR...")
    net = ie.read_network(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.warning("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.warning("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
   
    # Load network to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)
     
    #CLASSES = 11
    class_names = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]

    print("Preparing input blobs")
    
    # We define the batch size as x for easier use throughout
    # Note that if you want to have a different batch size, you would have to create different IR (see the Juypter Notebook
    # for more information on this)
    x = 110
    print("Batch size is {}".format(x))
        
    correct = 0
    wrong = 0
    total_inference = 0
    top_n = 0
    j = 0
    
    def run_it(start, n):
        #Setup an array to run inference on
        modulations = np.ndarray(shape=(x, 128, 1, 2))
        
        #Fill up the input array for this batch
        stop = start + x
        i = 0
        
        for item in X[start:stop]:
            modulations[i] = item.reshape([1,2,128]).transpose(2, 0, 1)
            i += 1

        # Loading model to the plugin    
        # Start inference
        infer_time = []

        t0 = time()
        res = exec_net.infer(inputs={input_blob: modulations})
        infer_time.append((time()-t0)*1000)

        # Processing output blob
        res = res[out_blob]
        
        #Check results for accuracy
        
        nonlocal correct
        nonlocal wrong
        nonlocal j
        
        # Function for calculating the amount correct, up to a certain "num"
        # For example, for num = 3, would return top_n, representing the amount of 
        # times the correct label was one of the top 3 probabilities predicted
        def top_n_accuracy(num):
            nonlocal top_n
            
            for i in range(num):
                det_label = class_names[top_ind[i]]

                if det_label == labels[j][0]:
                    top_n += 1
                    return
        
        #Automatically calculates the accuracy for top 1 predictions
        for i, probs in enumerate(res):
            probs = np.squeeze(probs)
            top_ind = np.argsort(probs)[-args.number_top:][::-1]
           
            det_label = class_names[top_ind[0]]
            
            if det_label == labels[j][0]:
                correct = correct + 1
            else:
                wrong = wrong + 1        
            
            #Default to calculating top-3 accuracy
            top_n_accuracy(3)
            
            j = j + 1        
            
        nonlocal total_inference
        total_inference += np.sum(np.asarray(infer_time))
        
        
    #Iterate through the whole dataset    
    num_batches = X.shape[0]//x
   
    k = 0
    n = 3
    print("Running inference: Batch 1")
    
    #Run it on the dataset
    for i in range(num_batches):
        if (i + 1) % 100 == 0:
            print("Running inference: Batch " + str(i + 1))
        run_it(k, n)
        k += x
         
    # Print results    
    print("Correct " + str(correct))
    print("Wrong " + str(wrong))
    print("Accuracy: " + str(correct/(correct + wrong)))
    print("Top " + str(n) + " Correct: " + str(top_n))
    print("Top " + str(n) + " Accuracy: " + str(top_n/(correct + wrong)))

    print("Average running time of one batch: {} ms".format(total_inference/num_batches))
    print("Total running time of inference: {} ms" .format(total_inference))
    print("Throughput: {} FPS".format((1000*args.number_iter*x*num_batches)/total_inference))
    
    import platform
    platform.processor()
    print("Processor: " + platform.processor())
    print("\n")
        
    #Cleanup
    del net
    del exec_net

if __name__ == '__main__':
    sys.exit(main() or 0) 