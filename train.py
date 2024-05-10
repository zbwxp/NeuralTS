from data_onehot import Bandit_onehot
from data_multi import Bandit_multi
from data_sanity import Bandit_sanity
from learner_linear_cuda import LinearTS
from learner_neural import NeuralTS
from learner_diag import NeuralTSDiag
from learner_kernel import KernelTS
from neural_boost import Boost
from learner_diag_kernel import KernelTSDiag
from learner_diag_linear import LinearTSDiag
import numpy as np
import argparse
import pickle 
import os
import time
import torch


if __name__ == '__main__':
    t1 = time.time()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    parser = argparse.ArgumentParser(description='Thompson Sampling')
    parser.add_argument('--encoding', default='multi', metavar='sanity|onehot|multi')

    parser.add_argument('--dim', default=100, type=int, help='dim for linear bandit, sanity only')
    parser.add_argument('--arm', default=10, type=int, help='arm for linear bandit, sanity only')
    parser.add_argument('--noise', default=1, type=float, help='noise for linear bandit, sanity only')
    parser.add_argument('--size', default=10000, type=int, help='bandit size')

    parser.add_argument('--dataset', default='mushroom', metavar='DATASET', help='encoding = onehot and multi only')
    parser.add_argument('--shuffle', type=bool, default=1, metavar='1 / 0', help='shuffle the data set or not, encoding = onehot and multi only')
    parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None, encoding = onehot and multi only')

    parser.add_argument('--r', type=float, default=0.9, metavar='r', help='ratio for feature norm, encoding = onehot only')

    parser.add_argument('--learner', default='linear', metavar='linear|neural|kernel|boost', help='TS learner')
    parser.add_argument('--inv', default='diag', metavar='diag|full', help='inverse matrix method')
    parser.add_argument('--style', default='ts', metavar='ts|ucb', help='TS or UCB')

    parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
    parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularzation')    
    parser.add_argument('--hidden', type=int, default=100, help='network hidden size, learner = neural and diag only')

    parser.add_argument('--p', type=float, default=0.8, help='p, learner = boost only')
    parser.add_argument('--q', type=int, default=5, help='q, learner = boost only')
    parser.add_argument('--delay', type=int, default=1, help='delay reward')
    parser.add_argument('--custom', type=bool, default=False, help='input the info of the custom dataset')
    parser.add_argument('--X_path', type=str, default='', help='path to the X dataset')
    parser.add_argument('--y_path', type=str, default='', help='path to the y dataset')
    parser.add_argument('--floatrwd', type=bool, default=True, help='use float reward or one hot reward')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='epoch for training')

    args = parser.parse_args()
    custom = {'X_path': args.X_path, 'y_path': args.y_path} if args.custom else None

    use_seed = None if args.seed == 0 else args.seed
    if args.encoding == 'sanity':
        b = Bandit_sanity(args.dim, args.noise, args.arm, args.size)
        bandit_info = 'sanity_{}_{}_{}_{}'.format(args.dim, args.noise, args.arm, args.size)
    elif args.encoding == 'onehot':
        b = Bandit_onehot(args.dataset, is_shuffle=args.shuffle, seed=use_seed, r=args.r)
        bandit_info = 'onehot_{}_{}'.format(args.dataset, args.r)
    elif args.encoding == 'multi':
        b = Bandit_multi(args.dataset, is_shuffle=args.shuffle, seed=use_seed, custom=custom)
        bandit_info = 'multi_{}'.format(args.dataset)
    else:
        raise RuntimeError('Encoding not exist')

    if args.learner == 'linear':
        if args.inv == 'diag':
            """ Linear TS diag is to use a cuda network """
            l = LinearTSDiag(b.dim, args.lamdba, args.nu, args.style)
        elif args.inv == 'full':
            l = LinearTS(b.dim, args.lamdba, args.nu, args.style)
        else:
            RuntimeError('Inverse method not exist')
        ts_info = '{}_linear_{:.3e}_{:.3e}_{}'.format(args.style, args.lamdba, args.nu, args.inv)
    elif args.learner == 'neural':
        print(f'args.dataset: {args.dataset}')
        if args.inv == 'diag':
            l = NeuralTSDiag(b.dim, args.lamdba, args.nu, args.hidden, args.style)
        elif args.inv == 'full':
            l = NeuralTS(b.dim, args.lamdba, args.nu, args.hidden, args.style)
        else:
            RuntimeError('Inverse method not exist')
        ts_info = '{}_neural_{:.3e}_{:.3e}_{}_{}'.format(args.style, args.lamdba, args.nu, args.hidden, args.inv)
    elif args.learner == 'kernel':
        if args.inv == 'diag':
            raise RuntimeError('Diag inverse estimation not feasible to kernel method!')
            l = KernelTSDiag(b.dim, args.lamdba, args.nu, args.style)
        elif args.inv == 'full':
           l = KernelTS(b.dim, args.lamdba, args.nu, args.style)
        else:
            RuntimeError('Inverse method not exist')
        ts_info = '{}_kernel_{:.3e}_{:.3e}_{}'.format(args.style, args.lamdba, args.nu, args.inv)
    elif args.learner == 'boost':
        l = Boost(b.dim, args.hidden, args.p, args.q)
        ts_info = 'boost_{:.1e}_{}_{}'.format(args.p, args.q, args.hidden)
    else:
        raise RuntimeError('Learner not exist')
    setattr(l, 'delay', args.delay)

    regrets = []
    regrets_0 = []
    regrets_1 = []
    regrets_2 = []
    regrets_3 = []
    regrets_4 = []
    regrets_5 = []
    regrets_6 = []

    accumulated_regret = []
    accumulated_regret_0 = []
    accumulated_regret_1 = []
    accumulated_regret_2 = []
    accumulated_regret_3 = []
    accumulated_regret_4 = []
    accumulated_regret_5 = []
    accumulated_regret_6 = []



    for t in range(min(args.size, b.size)):
        context, rwd = b.step(float_reward=args.floatrwd)
        arm_select, nrm, sig, ave_rwd = l.select(context)
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        reg0 = np.max(rwd) - rwd[0]
        reg1 = np.max(rwd) - rwd[1]
        reg2 = np.max(rwd) - rwd[2]
        reg3 = np.max(rwd) - rwd[3]
        reg4 = np.max(rwd) - rwd[4]
        reg5 = np.max(rwd) - rwd[5]
        reg6 = np.max(rwd) - rwd[6]
        regrets_0.append(reg0)
        regrets_1.append(reg1)
        regrets_2.append(reg2)
        regrets_3.append(reg3)
        regrets_4.append(reg4)
        regrets_5.append(reg5)
        regrets_6.append(reg6)

        accumulated_regret_0.append(np.sum(regrets_0))
        accumulated_regret_1.append(np.sum(regrets_1))
        accumulated_regret_2.append(np.sum(regrets_2))
        accumulated_regret_3.append(np.sum(regrets_3))
        accumulated_regret_4.append(np.sum(regrets_4))
        accumulated_regret_5.append(np.sum(regrets_5))
        accumulated_regret_6.append(np.sum(regrets_6))


        loss = l.train(context[arm_select], r, lr=args.lr, epoch=args.epochs)
        regrets.append(reg)
        if t % 100 == 0:
            print('{}: {:.3f}, {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(t, np.sum(regrets), loss, nrm, sig, ave_rwd))

        accumulated_regret.append(np.sum(regrets))
        # plot all the accumulated regrets
        if t == 399:
            import matplotlib.pyplot as plt
            # Assuming accumulated_regret and accumulated_regret_0 to accumulated_regret_6 are defined somewhere above this
            arr = np.array(accumulated_regret)  # Convert the list to a numpy array
            arr_0 = np.array(accumulated_regret_0)
            arr_1 = np.array(accumulated_regret_1)
            arr_2 = np.array(accumulated_regret_2)
            arr_3 = np.array(accumulated_regret_3)
            arr_4 = np.array(accumulated_regret_4)
            arr_5 = np.array(accumulated_regret_5)
            arr_6 = np.array(accumulated_regret_6)

            plt.figure(figsize=(10, 6))  # Size of the figure

            # Define colors for clarity
            colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown', 'pink', 'grey']

            # Plotting each array with a marker, each with its unique label and color
            plt.plot(np.arange(len(arr)), arr, label='Neural TS', marker='o', color=colors[0])
            plt.plot(np.arange(len(arr_0)), arr_0, label='Regret 0', marker='o', color=colors[1])
            plt.plot(np.arange(len(arr_1)), arr_1, label='Regret 1', marker='o', color=colors[2])
            plt.plot(np.arange(len(arr_2)), arr_2, label='Regret 2', marker='o', color=colors[3])
            plt.plot(np.arange(len(arr_3)), arr_3, label='Regret 3', marker='o', color=colors[4])
            plt.plot(np.arange(len(arr_4)), arr_4, label='Regret 4', marker='o', color=colors[5])
            plt.plot(np.arange(len(arr_5)), arr_5, label='Regret 5', marker='o', color=colors[6])
            plt.plot(np.arange(len(arr_6)), arr_6, label='Regret 6', marker='o', color=colors[7])

            # Adding a line from (0,0) to (len(arr)-1, max(arr)) with linestyle '--', in red, and with a specific label
            plt.plot([0, len(arr)-1], [0, max(arr)], label='Trend line', linestyle='--', color='red')

            plt.title('Array Visualization with Line')
            plt.xlabel('Index')
            plt.ylabel('Value')

            # Adding the legend outside the plot area
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.tight_layout()  # Adjust the layout to make room for the legend
            fig_name = f'{args.dataset}_{args.learner}_{args.inv}_{args.style}_{args.lamdba}_{args.nu}_{args.hidden}_{args.lr}_{args.epochs}.png'
            plt.savefig(fig_name)

        if t == 4000:
            import matplotlib.pyplot as plt
            arr = np.array(accumulated_regret)  # Convert the list to a numpy array
            plt.figure(figsize=(10, 6))  # Size of the figure

            # Plotting the numpy array
            plt.plot(np.arange(len(arr)), arr, label='Data', marker='o')  # Plotting the array

            # Drawing a straight line from (0,0) to (len(arr), max(arr))
            plt.plot([0, len(arr)-1], [0, max(arr)], label='Line from (0,0) to (len(arr), max(arr))', linestyle='--', color='red')

            # Adding titles and labels
            plt.title('Array Visualization with Line')
            plt.xlabel('Index')
            plt.ylabel('Value')
            # give a fig_name that contains the hyperparameters 
            fig_name = f'{args.dataset}_{args.learner}_{args.inv}_{args.style}_{args.lamdba}_{args.nu}_{args.hidden}_{args.lr}_{args.epochs}.png'
            plt.savefig(fig_name)

    filename = '{:.3f}_{}_{}_delay_{}_{}.pkl'.format(np.sum(regrets), bandit_info, ts_info, args.delay, time.time() - t1)
    with open(os.path.join('record', filename), 'wb') as f:
        pickle.dump(regrets, f)
