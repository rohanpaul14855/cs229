import numpy as np


def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    print(hdr)
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)


def laplace_smoothing(matrix, categories, tokenlist):
    '''
    :param matrix: 
    :param categories: 
    :return: 
    parameters is a n x 2 matrix where n corresponds to token and the 
    2 corresponds to 2 classes (not spam(0) and spam(1))
    '''
    categories = np.array(categories)
    parameters = np.zeros((matrix.shape[1], 2))

    #calculate priors
    p_spam = sum(categories)/len(categories)
    p_nonspam = (len(categories) - sum(categories))/len(categories)

    #calculate likelihoods for each token and class
    for y in [0, 1]:
        class_subset = matrix[categories == y]
        denom = class_subset.shape[0] + len(tokenlist)
        for token in range(matrix.shape[1]):
            parameters[token, y] = (sum(class_subset[:, token]) + 1)/denom
    return parameters, p_spam, p_nonspam


def nb_train(matrix, category):
    """
    :param matrix: 
    :param category: 
    :return: 
    parameters is a n x 2 matrix where n corresponds to token and the 
    2 corresponds to 2 classes (not spam(0) and spam(1))
    """
    state = {}
    N = matrix.shape[1]
    categories = np.array(category)
    parameters = np.zeros((N, 2))

    # calculate priors
    p_spam = sum(categories) / len(categories)
    p_nonspam = (len(categories) - sum(categories)) / len(categories)
    state['p_spam'] = p_spam
    state['p_nonspam'] = p_nonspam

    # calculate likelihoods for each token and class
    for y in [0, 1]:
        class_subset = matrix[categories == y]
        denom = np.sum(class_subset) + N    #sum all words in where label == y
        for token in range(N):
            parameters[token, y] = (sum(class_subset[:, token]) + 1) / denom
    spam_word_index = sorted(range(len(parameters)),
                             key=lambda x: [np.log(k[1]/k[0]) for k in parameters][x])[-5:]
    state['parameters'] = parameters

    return state, spam_word_index

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    # p(y = 1|x) =  [p(x|y=1)p(y=1)]/[p(x|y=1)p(y=1) + p(x|y=0)p(y=0)]
    p_spam = state['p_spam']
    p_nonspam = state['p_nonspam']
    parameters = state['parameters']
    for i in range(matrix.shape[0]):

        xs = [t for t, k in enumerate(matrix[i]) if k > 0]

        ######################################
        # p(y = 1|x)
        num = 0
        for j in xs:
            for _ in range(j):
                num += np.log(parameters[j, 1])
        num += np.log(p_spam)
        denom = num
        for j in xs:
            for _ in range(j):
                 denom += np.log(parameters[j, 0])
        denom += np.log(p_nonspam)
        prob_spam = num/denom

        ######################################
        # p(y = 0|x)

        num = 0
        for j in xs:
            for _ in range(j):
                num += np.log(parameters[j, 0])
        num += np.log(p_nonspam)
        denom = num
        for j in xs:
            for _ in range(j):
                denom += np.log(parameters[j, 1])
        denom += np.log(p_spam)
        prob_nonspam = num/denom

        print([prob_spam, prob_nonspam, prob_nonspam+prob_spam])
        output[i] = 0 if prob_spam > prob_nonspam else 1

    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')
    #Both have the same token list
    # print(tokenlist[1368], tokenlist[393], tokenlist[1356], tokenlist[1209], tokenlist[615])
    # exit()
    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return

if __name__ == '__main__':
    main()
