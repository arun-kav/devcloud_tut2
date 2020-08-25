# Store input arguments: <output_directory> <device> <fp_precision> <input_file>
cd $HOME/rf
OUTPUT_FILE=$1
DEVICE=$2
DTYPE=$3

# The default path for the job is the user's home directory,
#  change directory to where the files are.

# Make sure that the output directory exists.
mkdir -p $OUTPUT_FILEs

# Check for special setup steps depending upon device to be used
if [ "$DEVICE" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs - Updated for OpenVINO 2020.3
    export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2
    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
  #  export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
  
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/2020-4_PL2_FP11_AlexNet_GoogleNet_Generic.aocx

fi

# if [ "$DTYPE" = "FP16" ]; then
#     export MODEL=$HOME/rf/FP16rfoink.xml
    
# else
#     export MODEL=$HOME/rf/rfoink.xml
    
fi

python3 $HOME/rf/rf_tutorial/large_rf.py -m $MODEL -d $DEVICE

# Set inference model IR files using specified precision
# Run the classifier  code
