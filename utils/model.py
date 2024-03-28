import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import os
import subprocess

def print_graph_to_file(checkpoint_dir):
    out_file = open(checkpoint_dir + 'graph_printout.txt', 'w')
    for op in tf.get_default_graph().get_operations():
        out_file.write("%s\n" % (str(op.name)))
    out_file.close()

def freeze(exp_id, model_id_final, out_nodes):

    tf.reset_default_graph()

    input_graph_path = '/mnt/SSD_1/checkpoints/' + exp_id + '/input_graph.pb'
    input_saver_def_path = ""
    input_binary = False
    input_checkpoint_path = '/mnt/SSD_1/checkpoints/' + exp_id + '/' + model_id_final

    output_nodes_string = ''
    for i in range(len(out_nodes)):
        if i == (len(out_nodes) - 1):
            output_nodes_string = output_nodes_string + out_nodes[i]
        else:
            output_nodes_string = output_nodes_string + out_nodes[i] + ','

    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = '/mnt/SSD_1/checkpoints/' + exp_id + '/' + exp_id + '_graph.pb'
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, input_checkpoint_path,
                              output_nodes_string, restore_op_name,
                              filename_tensor_name, output_graph_path,
                              clear_devices,'')

    output_scrambled_path = '/mnt/SSD_1/checkpoints/' + exp_id + '/' + exp_id 
    generate_scrambled_proto(output_graph_path, output_scrambled_path)
    os.remove(output_graph_path)


def restore_session(checkpoint_dir, saver, session, iter=None):
    global_step = 0
    if not os.path.exists(checkpoint_dir):
        raise IOError(checkpoint_dir + ' does not exist.')
    else:
        path = tf.train.get_checkpoint_state(checkpoint_dir)
        if path is None:
            raise IOError('No checkpoint to restore in ' + checkpoint_dir)
        else:
            path_prefix = path.model_checkpoint_path.split('-')[0]
            path_iter = path.model_checkpoint_path.split('-')[1]
            if iter is not None:
                # restoring the model from iteration 'iter'
                load_path = path_prefix + '-' + iter
                global_step = int(iter)
            else:
                # restoring the model from the most recent iteration
                load_path = path_prefix + '-' + path_iter
                global_step = int(path_iter)

            print('Loading:', load_path)
            saver.restore(session, load_path)
            
    return global_step


def set_train_folder(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        while True:
            command = input('\nFolder ' + checkpoint_dir + ' already exists. Overwrite? [y/n]: ')
            if command == 'y':
                subprocess.run(["rm", "-r", checkpoint_dir])
                subprocess.run(["mkdir", checkpoint_dir])
                subprocess.run(["mkdir", checkpoint_dir + 'train/'])
                break
            else:
                print('\nPlease delete Folder ' + checkpoint_dir + ' manually or enter a different experiment ID')
                exit()
    else:
        subprocess.run(["mkdir", checkpoint_dir])
        subprocess.run(["mkdir", checkpoint_dir + 'train/'])

    #self.train_writer = tf.summary.FileWriter(self.checkpoint_dir + 'train/', self.model.session.graph)
    #tf.train.write_graph(self.model.session.graph_def, self.checkpoint_dir, 'input_graph.pb')


def set_val_folder(checkpoint_dir, dataset):
    if os.path.isdir(checkpoint_dir + 'eval_' + dataset):
        while True:
            command = input('\nFolder ' + checkpoint_dir + 'eval_' + dataset + ' already exists. Overwrite? [y/n]: ')
            if command == 'y':
                subprocess.run(["rm", "-r", checkpoint_dir + 'eval_' + dataset])
                subprocess.run(["mkdir", checkpoint_dir + 'eval_' + dataset])
                break
            else:
                print('Please delete/rename Folder ' + checkpoint_dir + 'eval_' + dataset + ' manually')
                exit()
    else:
        subprocess.run(["mkdir", checkpoint_dir + 'eval_' + dataset])

    #self.val_writer = tf.summary.FileWriter(self.checkpoint_dir + 'eval_' + self.dataset, self.model.session.graph)


def activation_summary(name, x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    tf.summary.histogram(name + '/activations', x)
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))


def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.stack(output_list)

def unpool_layer2x2(x, raveled_argmax, out_shape, name):
    with tf.name_scope(name):
        argmax = unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
        output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

        t2 = tf.squeeze(argmax)
        t2 = tf.stack((t2[0], t2[1]), axis=0)
        t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

        t = tf.concat(axis=3, values=[t2, t1])
        indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

        x1 = tf.squeeze(x)
        x1 = tf.reshape(x1, [-1, channels])
        x1 = tf.transpose(x1, perm=[1, 0])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

def unpool_layer2x2_batch(x, argmax, name):
    '''
    Args:
        x: 4D tensor of shape [batch_size x height x width x channels]
        argmax: A Tensor of type Targmax. 4-D. The flattened indices of the max
        values chosen for each output.
    Return:
        4D output tensor of shape [batch_size x 2*height x 2*width x channels]
    '''
    with tf.name_scope(name):
        x_shape = tf.shape(x)
        out_shape = [x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]]

        batch_size = out_shape[0]
        height = out_shape[1]
        width = out_shape[2]
        channels = out_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        argmax = unravel_argmax(argmax, argmax_shape)

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
        t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

        t2 = tf.to_int64(tf.range(batch_size))
        t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
        t2 = tf.reshape(t2, [-1, batch_size])
        t2 = tf.transpose(t2, perm=[1, 0])
        t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

        t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

        t = tf.concat(axis=4, values=[t2, t3, t1])
        indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

        x1 = tf.transpose(x, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(out_shape))
        return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))

def generate_scrambled_proto(input_path, output_path):
    '''
    Args:
        input_path: Location of frozen proto file
        output_path: Destination path to write the scrambled proto file
    '''
    with open(input_path, 'rb') as f:
        content = f.read()
    f.close()

    mod = 0
    mod_3 = 0

    buf = []

    for i in range(0,len(content)):
        if ( mod_3 == 0 ):
            buf.append(((content[i]) + mod) % 256 )
        elif (mod_3 == 1):
            buf.append(((content[i]) - mod) % 256 )
        else:
            buf.append(((content[i]) + mod + mod_3) % 256 )
        mod += 1
        mod_3 += 1
        if (mod == 256):
            mod = 0
        if (mod_3 == 3):
            mod = 0

    with open(output_path,"wb") as f:
        f.write(bytes(buf))
    f.close()

def generate_unscrambled_proto(input_path, output_path):
    '''
    Args:
        input_path: Location of scrambled proto file
        output_path: Destination path to write the unscrambled proto file
    '''
    with open(input_path, 'rb') as f:
        content = f.read()
    f.close()

    mod = 0
    mod_3 = 0

    buf = []

    for i in range(0,len(content)):
        if ( mod_3 == 0 ):
            buf.append(((content[i]) - mod) % 256 )
        elif (mod_3 == 1):
            buf.append(((content[i]) + mod) % 256 )
        else:
            buf.append(((content[i]) - mod - mod_3) % 256 )
        mod += 1
        mod_3 += 1
        if (mod == 256):
            mod = 0
        if (mod_3 == 3):
            mod = 0

    with open(output_path,"wb") as f:
        f.write(bytes(buf))
    f.close()
