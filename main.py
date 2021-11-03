import glob
import time

import tensorflow as tf
from sklearn.manifold import TSNE
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.models import Model
from tqdm import tqdm

# from SpectralClustering import testSpec
from load_data import load_data
from metrics import compute_and_print_scores

print(tf.__version__)
os.environ['CUDA_VISIBLE_DEVICES']='0'


def generator_encoder(input_dimension, output_dimension = 128):
    model = keras.Sequential()
    model.add(layers.Dense(128, input_shape=(input_dimension,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # model.add(layers.Dense(512))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(output_dimension, activation='tanh'))
    model.add(layers.BatchNormalization())

    return model


def discriminator_model(input_dim = 128):
    model = keras.Sequential()

    model.add(layers.Dense(64, input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # model.add(layers.Dense(256))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    model.add(layers.Dense(1))
    model.add(Activation('sigmoid'))

    return model


def AttentionModule(out_weight_dimension, input_dimension = 128):
    """
    the input of this module is the concatenated features h from{h1,h2...hn}
    and the output is a V dimensional weight vector w

    """
    model = keras.Sequential()

    model.add(layers.Dense(64, input_shape=(input_dimension * out_weight_dimension, )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # model.add(layers.Dense(256))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    model.add(layers.Dense(out_weight_dimension))
    model.add(layers.BatchNormalization())

    model.add(Activation('sigmoid'))
    #没有用τ
    model.add(layers.Softmax())

    return model


def Conv1():
    pass


def ClusteringModule(num_types, input_dimension = 128):
    fused_fea = Input((input_dimension,), name='input')
    dense1_out = layers.Dense(64)(fused_fea)
    relu_out1 = layers.ReLU()(dense1_out)
    batch_norm_out1 = layers.BatchNormalization()(relu_out1)
    # dense2_out = layers.Dense(256)(batch_norm_out1)
    # relu_out2 = layers.ReLU()(dense2_out)
    # batch_norm_out2 = layers.BatchNormalization()(relu_out2)
    dense3_out = layers.Dense(num_types)(batch_norm_out1)
    softmax_out = layers.Softmax()(dense3_out)

    model = Model(inputs=fused_fea, outputs=[relu_out1, softmax_out])

    # model = keras.Sequential()
    #
    # model.add(layers.Dense(256, input_shape=(input_dimension,)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.ReLU())
    #
    # model.add(layers.Dense(num_types))
    #
    # model.add(layers.Softmax())

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)


def discriminator_loss(real_latentRepre_out, fake_latentRepre_out, lambda_dis=1):
    real_loss = cross_entropy(tf.ones_like(real_latentRepre_out),real_latentRepre_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_latentRepre_out),fake_latentRepre_out)
    return lambda_dis * (real_loss + fake_loss)


def generator_loss(fake_latentRepre_out, lambda_gen=1):
    return lambda_gen * cross_entropy(tf.ones_like(fake_latentRepre_out),fake_latentRepre_out)


def vector_square_distance(vec1, vec2, mode = 1):
    """没用"""
    dis = 0.0
    for i in range(0, vec1.shape[0]):
        dis = dis + (vec1[i] - vec2[i])**2
    if mode == 1:
        return dis
    elif mode == 2:
        return tf.sqrt(dis)


def init_zeros_matrix(height, width):
    """没用"""
    return_matrix = []
    for i in range(0, height):
        tmp = []
        for j in range(0, width):
            tmp.append(0.)
        return_matrix.append(tmp)
    return return_matrix


def Gaussian_Kernel_Matrix(latent_representation):
    """
    input one view's latent representation matrix to compute ???, and output the K matrix
    return numpy array Gaussian_Kernel_Matrix

    """
    innerProduct = tf.matmul(latent_representation, tf.transpose(latent_representation))
    diag = tf.matmul(tf.reshape(tf.linalg.diag_part(innerProduct), (-1, 1)), tf.ones([1, latent_representation.shape[0]]))
    dist = diag - 2 * innerProduct + tf.transpose(diag)
    sigma = 0.15 * tf.reduce_mean(dist) * (latent_representation.shape[0] / (latent_representation.shape[0] - 1))
    K = tf.exp(-dist / sigma)

    return K


def attention_loss(Kf, Kc, lambda_att):
    """F范数"""
    return lambda_att * tf.norm(Kf - Kc)


def xTKx(vector_column1, Kf, vector_column2):
    """没用"""
    vector_column1 = tf.reshape(vector_column1, (1,-1))
    vector_column2 = tf.reshape(vector_column2, (1,-1))
    dot_pre = tf.matmul(vector_column1, Kf)
    dot = tf.matmul(dot_pre, tf.transpose(vector_column2))
    dot = tf.reshape(dot, [])
    return dot


def strict_upper_triangle_sum(tmp):
    diag_tmp = tf.linalg.diag_part(tmp)
    diag_tmp_M = tf.linalg.diag(diag_tmp)
    triu = tf.linalg.band_part(tmp, 0, -1) - diag_tmp_M
    val_sum = tf.reduce_sum(triu)
    return val_sum


def cs_divergence(clustering_result, Kf):
    H = tf.matmul(tf.matmul(tf.transpose(clustering_result), Kf), clustering_result)
    diagH = tf.reshape(tf.linalg.diag_part(H), (-1, 1))
    tmp = (H + 1e-9)/(tf.sqrt(tf.matmul(diagH, tf.transpose(diagH))) + 1e-9)
    val = strict_upper_triangle_sum(tmp)
    return val


def cluster_minus_simplex(clustering_result):
    innerProduct = tf.matmul(clustering_result, tf.transpose(clustering_result))
    diag = tf.matmul(tf.reshape(tf.linalg.diag_part(innerProduct), (-1, 1)),
                     tf.ones([1, clustering_result.shape[1]]))
    dist = diag - 2 * clustering_result + 1
    B = tf.exp(-dist)
    return B


def Pij(clustering_result):
    numerator = clustering_result ** 2 / tf.reduce_sum(clustering_result, 0)
    denominator = tf.reduce_sum(numerator, 1)
    return tf.transpose(tf.transpose(numerator) / denominator)


def clustering_loss(clustering_result, Kf, lambda_clu=1):
    c_m_simplex_M = cluster_minus_simplex(clustering_result)
    return lambda_clu * (strict_upper_triangle_sum(tf.matmul(tf.transpose(clustering_result), clustering_result)) + \
          cs_divergence(clustering_result, Kf) + cs_divergence(c_m_simplex_M, Kf))
    # KLD = 0.0
    # p = Pij(clustering_result)
    # for i in range(clustering_result.shape[0]):
    #     for j in range(clustering_result.shape[1]):
    #         KLD += p[i][j] * tf.math.log(p[i][j] / clustering_result[i][j])
    #return lambda_clu * (KLD)


def list_based_matrix_scale_up(matrix, weight):
    """没用"""
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = matrix[i][j] * weight
    return matrix


def train_step(dataset_batch):

    """will not update the i_th discriminator"""
    for i in range(0, num_modalities):
        for i_repeat in range(1):
            with tf.GradientTape(persistent=True) as generator1_tape, tf.GradientTape() as discriminator1_tape\
                    , tf.GradientTape(persistent=True) as generator2_tape, tf.GradientTape() as discriminator2_tape\
                    , tf.GradientTape(persistent=True) as generator3_tape, tf.GradientTape() as discriminator3_tape\
                    , tf.GradientTape(persistent=True) as generator4_tape, tf.GradientTape() as discriminator4_tape\
                    , tf.GradientTape(persistent=True) as generator5_tape, tf.GradientTape() as discriminator5_tape\
                    , tf.GradientTape(persistent=True) as attention_tape, tf.GradientTape() as cluster_module_tape:

                # t1 = time.perf_counter()
                input_feature = []
                for j in range(num_modalities):
                    input_feature.append(dataset_batch[str(j+1)])

                """generator_out are latent representations"""
                generator_out = []
                for j in range(0, num_modalities):
                    out = generator_encoders[j](input_feature[j], training=True)
                    if j == 0:
                        concated_out = out
                    else:
                        concated_out = tf.concat([concated_out, out], 1)
                    generator_out.append(out)

                """output of discriminator"""
                gen_loss = []
                discrimi_loss = []
                for j in range(0, num_modalities):
                    gen_loss.append(tf.constant(0.0))
                    discrimi_loss.append(tf.constant(0.0))
                    if j != i:
                        """real out is the output of reference latent representation, while fake ones are others"""
                        real_out = discriminators[i](generator_out[i], training=True)
                        fake_out = discriminators[i](generator_out[j], training=True)
                        discrimi_loss[j] = discriminator_loss(real_out, fake_out,LAMB_DIS)
                        gen_loss[j] = generator_loss(fake_out, LAMB_GEN)

                attention_weight_out = attention_module(concated_out)
                mean_weight = tf.reduce_mean(attention_weight_out, 0)

                gaussian_matrix = []
                for j in range(num_modalities):
                    gaussian_matrix.append(Gaussian_Kernel_Matrix(generator_out[j]))

                """get the fused representation and gaussian matrix with the mean weight and generator out"""
                for i_th_weight in range(num_modalities):
                    weight = mean_weight[i_th_weight]
                    if i_th_weight==0:
                        fused_representation = generator_out[i_th_weight] * weight
                        Kc = gaussian_matrix[i_th_weight] * weight
                    else:
                        fused_representation = fused_representation + generator_out[i_th_weight] * weight
                        Kc = Kc + gaussian_matrix[i_th_weight] * weight

                hidden_cluster_repre, clustering_result = cluster_module(fused_representation)
                Kf = Gaussian_Kernel_Matrix(hidden_cluster_repre)

                att_loss = attention_loss(Kf, Kc, lambda_att=LAMB_ATT)
                cluster_loss = clustering_loss(clustering_result, Kf, lambda_clu=LAMB_CLU)

                # print('T1:')
                # print(f'{time.perf_counter()-t1 : .4f}')

            # t2 = time.perf_counter()
            """
            更新梯度
            """
            gen_tapes = [generator1_tape, generator2_tape, generator3_tape, generator4_tape, generator5_tape]
            disc_tapes = [discriminator1_tape, discriminator2_tape, discriminator3_tape, discriminator4_tape, discriminator5_tape]
            
            discgrad_disc = disc_tapes[i].gradient(discrimi_loss[i], discriminators[i].trainable_variables)
            discriminator_optimizer.apply_gradients(zip(discgrad_disc, discriminators[j].trainable_variables))

            for j in range(num_modalities):
                if j != i:
                    gengrads_gen = gen_tapes[j].gradient(gen_loss[j], generator_encoders[j].trainable_variables)
                    generator_optimizer.apply_gradients(zip(gengrads_gen, generator_encoders[j].trainable_variables))
                gengrads_att = gen_tapes[j].gradient(att_loss, generator_encoders[j].trainable_variables)
                gengrads_clus = gen_tapes[j].gradient(cluster_loss, generator_encoders[j].trainable_variables)
                generator_optimizer.apply_gradients(zip(gengrads_att, generator_encoders[j].trainable_variables))
                generator_optimizer.apply_gradients(zip(gengrads_clus, generator_encoders[j].trainable_variables))
            gradient_att_1 = attention_tape.gradient(att_loss, attention_module.trainable_variables)
            gradient_att_2 = attention_tape.gradient(cluster_loss, attention_module.trainable_variables)
            gradient_cluster = cluster_module_tape.gradient(cluster_loss, cluster_module.trainable_variables)

            cluster_optimizer.apply_gradients(zip(gradient_cluster, cluster_module.trainable_variables))
            attention_optimizer.apply_gradients(zip(gradient_att_1, attention_module.trainable_variables))
            attention_optimizer.apply_gradients(zip(gradient_att_2, attention_module.trainable_variables))

            del generator1_tape, generator2_tape, generator3_tape, attention_tape

        # print('T2:')
        # print(f'{time.perf_counter() - t2 : .4f}')

    whole_loss = 0.0
    gen_all = 0.0
    disc_all = 0.0
    for j in range(num_modalities):
        whole_loss = whole_loss + gen_loss[j] + discrimi_loss[j]
        gen_all = gen_all + gen_loss[j]
        disc_all = disc_all + discrimi_loss[j]
    whole_loss = whole_loss + att_loss
    whole_loss = whole_loss + cluster_loss
    return whole_loss, gen_all, disc_all, att_loss, cluster_loss, clustering_result.numpy()


def save_model(model_path):
    for i in range(num_modalities):
        generator_encoders[i].save(model_path + 'generator' + str(i) + '.h5')
        discriminators[i].save(model_path + 'discrimi' + str(i) + '.h5')
    attention_module.save(model_path + 'attention.h5')
    cluster_module.save(model_path + 'cluster.h5')


def make_model(model_path, num_modalities):
    generator_encoders = []
    discriminators = []

    for i in range(0, num_modalities):
        model_path_generator = model_path + 'generator' + str(i) + '.h5'
        model_path_discrimi = model_path + 'discrimi' + str(i) + '.h5'
        if not os.path.exists(model_path_generator):
            feature_dimension_i = fea[str(i + 1)].shape[1]
            print(feature_dimension_i)
            generator_encoders.append(generator_encoder(feature_dimension_i, output_dimension=LATENT_DIM))
            discriminators.append(discriminator_model(input_dim=LATENT_DIM))
        else:
            generator_encoders.append(keras.models.load_model(model_path_generator))
            discriminators.append(keras.models.load_model(model_path_discrimi))

    if not os.path.exists(model_path + 'attention.h5'):
        attention_module = AttentionModule(input_dimension=LATENT_DIM, out_weight_dimension=num_modalities)
    else:
        attention_module = keras.models.load_model(model_path + 'attention.h5')
    if not os.path.exists(model_path + 'cluster.h5'):
        cluster_module = ClusteringModule(input_dimension=LATENT_DIM, num_types=num_classes)
    else:
        cluster_module = keras.models.load_model(model_path + 'cluster.h5')

    return generator_encoders, discriminators, attention_module, cluster_module


def delete_all():
    os.remove('scores.txt')
    os.remove('tst_sco.txt')
    os.remove('out.txt')
    files = glob.glob('./save_weights/*')
    for file in files:
        os.remove(file)


def add_all_list_items(lbs, added_lbs_list):
    """没用"""
    for lb in added_lbs_list:
        lbs.append(lb)
    return lbs


def tsne_visual(data, labels, name):
    tsne = TSNE(init='pca', random_state=0)
    result = tsne.fit_transform(data)
    fig = plt.figure()
    ax = plt.subplot(111)
    print(result.shape[0])
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    shape = ['.', '^', '*', 'p', 's']
    formation = []
    for i in range(8):
        for j in range(5):
            tmp = color[i] + shape[j]
            formation.append(tmp)
    for i in range(result.shape[0]):
        plt.plot(result[i, 0], result[i, 1], formation[labels[i]-1])
    plt.xticks([])
    plt.yticks([])
    plt.title(name)
    plt.show()


def test(test_dataset):
    """generator_out are latent representations"""
    generator_out = []
    input_feature = []

    for batch in test_dataset:
        test_dataset = batch

    for j in range(num_modalities):
        input_feature.append(test_dataset[str(j + 1)])

    for j in range(0, num_modalities):
        out = generator_encoders[j](input_feature[j], training=False)
        if j == 0:
            concated_out = out
        else:
            concated_out = tf.concat([concated_out, out], 1)
        generator_out.append(out)

    attention_weight_out = attention_module(concated_out, training=False)
    mean_weight = tf.reduce_mean(attention_weight_out, 0)
    print(mean_weight)
    for i_th_weight in range(num_modalities):
        weight = mean_weight[i_th_weight]
        if i_th_weight == 0:
            fused_representation = generator_out[i_th_weight] * weight
        else:
            fused_representation = fused_representation + generator_out[i_th_weight] * weight
    hidden_cluster_repre, clustering_result = cluster_module(fused_representation, training=False)

    generator_out = np.array(generator_out)
    print(len(generator_out))
    print(len(generator_out[0]))
    hidden_cluster_repre = np.array(hidden_cluster_repre)
    print(len(hidden_cluster_repre))
    lbs = test_dataset['label']
    print(len(lbs))
    # tsne_visual(hidden_cluster_repre, lbs, 'tsne')
    # tsne_visual(generator_out[1], lbs, 'tsne1')

    sco = compute_and_print_scores(clustering_result, test_dataset['label'].numpy().tolist(), mode='test')

    f_test = open('tst_sco.txt', 'a+')
    f_test.write(str(sco))
    f_test.write('\n')

    return sco


def train(dataset, epochs):
    global now_highest
    for epoch in range(epochs):

        # 创建一个进度条
        with tqdm(total=len(dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='it', colour='white') as pbar:
            for dataset_batch in dataset:
                whole_loss, gen_all, disc_all, att_loss, cluster_loss, cluster_result = train_step(dataset_batch)
                pbar.set_postfix({'batch_loss' : whole_loss.numpy(), 'cluster_loss' : cluster_loss.numpy()})
                out = open("out.txt", "a+")
                out.write('batch_loss : ' + str(whole_loss.numpy()) + '  gen_all : ' + str(gen_all.numpy()) +
                          '  disc_all : ' + str(disc_all.numpy()) + '  att_loss : ' + str(att_loss.numpy()) +
                          '  cluster_loss : ' + str(cluster_loss.numpy()) + '\n')
                out.close()

                sco = compute_and_print_scores(cluster_result, dataset_batch['label'].numpy().tolist(), mode='train')

                pbar.update(1)
                save_model(model_path)
                

        dataset = tf.data.Dataset.shuffle(dataset, len(dataset))


now_highest = 0.84

num_modalities = 2
LATENT_DIM = 64

generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
attention_optimizer = tf.keras.optimizers.Adam(1e-4)
cluster_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCH = 15
datapath = 'convert_data/MNIST_2views.mat'
num_classes = 10
TEST_BATCH_SIZE = 500
BATCH_SIZE = 100
dataset, test_dataset, fea = load_data(datapath, num_modalities, TEST_BATCH_SIZE, BATCH_SIZE)

LAMB_DIS = 1
LAMB_GEN = 24
LAMB_ATT = 3
LAMB_CLU = 1.5

model_path = './save_weights/'
generator_encoders, discriminators, attention_module, cluster_module = make_model(model_path, num_modalities)

# print('My model:')
# test(test_dataset)
# print('Compared model:')
# testSpec(test_dataset, num_classes, num_modalities)
for i_epo in range(5):
    train(dataset, EPOCH)
    # acctmp = test(test_dataset)
    # if acctmp>now_highest:
    #     save_model('./best_weights/now_highest/')
    #     now_highest = acctmp
#test(test_dataset)
