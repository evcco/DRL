#########################
## Author: Chaochao Lu ##
#########################
import vanilla_config as configs
from model_con import Model_Con
from data_handler import DataHandler

from con_ac_train import *

def main():
    opts = configs.model_config
    opts['qzgz_net_layers'] = [128, 128]  # Example layer sizes
    opts['qzgz_mu_outlayers'] = [64, 64]  # Example layer sizes

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    gpu_config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=False, log_device_placement=False)
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)

    print('starting processing data ...')

    data = DataHandler(opts)

    print('starting initialising model ...')
    opts['r_range_upper'] = data.train_r_max
    opts['r_range_lower'] = data.train_r_min
    model = Model_Con(sess, opts)

    opts['batch_size'] = 1
    opts['va_sample_num'] = 6
    opts['model_bn_is_training'] = False

    print('starting training policy using AC_Con ...')
    ac = AC_Con(sess, opts, model)
    ac.train(data)

if __name__ == "__main__":
    main()