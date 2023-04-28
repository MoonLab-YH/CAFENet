import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

from trainer import TapLab
from data.generator_FSE import FSE_generator
from utils.Heuristics import Roataion_Inference
from utils.functions import *
from utils.evaluator import Evaluator
from tqdm import tqdm
from tensorboardX import SummaryWriter

def DeletePrev():
    # raise ValueError
    DeleteContent('summary/eval_loss')
    DeleteContent('summary/train_loss')
    DeleteContent('output/testset')
    DeleteContent('output/trainset')
    DeleteContent('model_save')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--mode', type=str, default='training', choices=['training', 'testing'])
    parser.add_argument('--pretrain_iter', type=int, default=40000)

    torch.set_num_threads(1)
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False

    max_iter = 40003
    eval_max_iter = 150
    test_max_iter = 10000

    lr_rate = 0.0001
    lrstep = 38000
    gamma = 0.1
    n_shot = 5
    nb_class_train = 1
    nb_samples = 10
    n_query = nb_samples - n_shot
    train_dir = '/drive1/YH/datasets/FSE-1000/fewshot_data/Train/'
    eval_dir = '/drive1/YH/datasets/FSE-1000/fewshot_data/Test/'
    pretrain_iter = args.pretrain_iter

    eval_interval = 5000
    show_interval = 50
    save_interval = 100
    model_save_interval = 2500
    test_show_interval = 20
    test_save_interval = 20

    # ------------------
    model = TapLab(nb_class=1, input_size=320, dropout_aspp=0.2, n_shot=n_shot, dropout_dec=[0.2, 0.2], device='cuda')
    model.set_optimizer(learning_rate=lr_rate, weight_decay_rate1=3e-4, weight_decay_rate2 = 3e-4, lrstep=lrstep, decay_rate=gamma)
    if args.cuda:
        model = torch.nn.DataParallel(model)
        model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print('total_param ; %d' % total_params)
    print('Mode : %s '%(args.mode))

    if args.mode == 'training':
        print('Train!')
        save_cnt = 0

        train_writer = SummaryWriter('summary/train_loss')
        eval_writer = SummaryWriter('summary/eval_loss')

        train_generator = FSE_generator(path_dir=train_dir, n_epoch=max_iter, train=True)

        loss_TAPs = []; loss_edges = []; loss_dices = []

        for t, images, labelSeg, GT in tqdm(train_generator):

            if args.cuda:
                images, labelSeg, GT = images.cuda(), labelSeg.cuda(), GT.cuda()

            loss_TAP, loss_edge, loss_dice = model.module.Train(images=images, labels=labelSeg, GT=GT)
            loss_TAPs.append(loss_TAP);
            loss_edges.append(loss_edge)
            loss_dices.append(loss_dice)

            if (t % show_interval == 0 ):
                mean_TAP = sum(loss_TAPs) / len(loss_TAPs)
                mean_edge = sum(loss_edges) / len(loss_edges)
                mean_dice = sum(loss_dices) / len(loss_dices)

                print("\n[Train] Episode: %d, loss_TAP: %f, loss_edge: %f, loss_dice: %f" % (t, mean_TAP, mean_edge,  mean_dice))
                train_writer.add_scalar('loss_TAP', mean_TAP, t)
                train_writer.add_scalar('loss_edge', mean_edge, t)
                train_writer.add_scalar('total_dice', mean_dice, t)
                loss_TAPs = []; loss_edges = []; loss_dices = [];

            if (t != 0) and (t % eval_interval == 0):
                eval_loss_TAPs = [];  eval_loss_edges=[]; eval_loss_dices=[]; eval_APs=[]
                test_generator = FSE_generator(
                    path_dir=eval_dir, n_epoch=eval_max_iter,train=False)

                for i, images, labelSeg, GT in test_generator:
                    if args.cuda:
                        images, labelSeg, GT = images.cuda(), labelSeg.cuda(), GT.cuda()

                    loss_TAP, loss_edge, loss_dice, final_output, pred_TAP, AP \
                        = model.module.evaluate(images=images, labels=labelSeg, GT=GT)

                    eval_loss_TAPs.append(loss_TAP); eval_loss_edges.append(loss_edge)
                    eval_loss_dices.append(loss_dice); eval_APs.append(AP)

                eval_mean_TAP = sum(eval_loss_TAPs) / len(eval_loss_TAPs)
                eval_mean_edge = sum(eval_loss_edges) / len(eval_loss_edges)
                eval_mean_dice = sum(eval_loss_dices) / len(eval_loss_dices)
                eval_AP = sum(eval_APs) / len(eval_APs)

                print("[Evaluation] Episode: %d, loss_TAP: %f, loss_edge : %f, loss_dice: %f, AP : %f"%
                      (t, eval_mean_TAP, eval_mean_edge, eval_mean_dice, eval_AP))

                eval_writer.add_scalar('loss_TAP', eval_mean_TAP, t)
                eval_writer.add_scalar('loss_edge', eval_mean_edge, t)
                eval_writer.add_scalar('loss_dice', eval_mean_dice, t)
                eval_writer.add_scalar('AP', eval_AP, t)


            if t % save_interval == 0 and t != 0:
                test_generator = FSE_generator(
                    path_dir=eval_dir, n_epoch=1, train=False)
                dir='testset'
                save_cnt = (save_cnt + 1) % 6

                for i, images, labelSeg, GT in test_generator:
                    if args.cuda:
                        images, labelSeg, GT = images.cuda(), labelSeg.cuda(), GT.cuda()

                    _,_,_,final_output,pred_TAP,AP = model.module.evaluate(images=images, labels=labelSeg, GT=GT)

                    save_path = 'output/' + dir + '/' + str(save_cnt) + '_'
                    for idx in range(5):
                        visualize(images[5 + idx], save_path + str(idx) + '_img.jpg', color=True)
                        visualize(labelSeg[5 + idx], save_path + str(idx) + '_Seg.jpg')
                        visualize(GT[5 + idx], save_path + str(idx) + '_GT.jpg')
                        visualize(final_output[idx], save_path + str(idx) + '_pred.jpg')
                        visualize(final_output[idx], save_path + str(idx) + '_threshold_pred.jpg', threshold=200)
                        visualize(pred_TAP[idx, 1], save_path + str(idx) + '_TAP.jpg')

                print('image saved...!! %s %d'%(dir,save_cnt))


            if t % model_save_interval == 0 and t != 0:
                model_path = 'model_save'
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                torch.save(model.state_dict(), os.path.join(model_path,str(t) + '.pth'))
                print('model parameter at iter %d is saved' % t)

    elif args.mode == 'testing':
        print('Pretrain Iter : %d'%(args.pretrain_iter))
        DeleteContent('output/testset')
        DeleteContent('output/trainset')
        DeleteContent('output/support')
        model_path = 'model_save'
        model.load_state_dict(torch.load(os.path.join(model_path, str(pretrain_iter) + '.pth')))
        model.eval()

        save_cnt = 0
        evaluator = Evaluator()
        test_loss_TAPs = []; test_loss_edges = []; test_loss_dices = []; test_APs = []; test_AP_original =[]
        AP_list = []


        test_generator = FSE_generator(
            path_dir=eval_dir, n_epoch=test_max_iter, n_shot=n_shot, train=False)

        for epoch, images, labelSeg, GT in tqdm(test_generator):

            images, labelSeg, GT = images.cuda(), labelSeg.cuda(), GT.cuda()
            loss_TAP, loss_edge, loss_dice, final_output, pred_TAP, AP \
                = model.module.evaluate(images=images, labels=labelSeg, GT=GT)
            GT = Binarize(GT, threshold=100)[n_shot:]
            for idx in range(len(final_output)):
                evaluator.add_batch(final_output[idx], GT[idx])
            AP_list.append(AP)

            if (epoch) % test_show_interval == 0:
                print(f'test episode : {epoch}: AP: {np.mean(AP_list).mean()}, F1: {evaluator.F1_measure().max()}')

            if epoch % test_save_interval == 0:
                test_generator = FSE_generator(
                    path_dir=eval_dir, n_epoch=1, train=False)
                dir = 'testset'
                save_cnt = (save_cnt + 1)% 6

                for i, images, labelSeg, GT in test_generator:
                    final_output, AP, original_AP, pred_TAP = Roataion_Inference(images, labelSeg, GT, model)

                    save_path = 'output/' + dir + '/' + str(save_cnt) + '_'
                    size = 320 if dir == 'trainset' else 512

                    for idx in range(5):
                        visualize(images[5 + idx], save_path + str(idx) + '_img.jpg', color=True, size=320)
                        visualize(labelSeg[5 + idx], save_path + str(idx) + '_Seg.jpg', size=None)
                        visualize(GT[5 + idx], save_path + str(idx) + '_GT.jpg', size=None)
                        visualize(final_output[idx], save_path + str(idx) + '_pred.jpg', size=None)
                        visualize(final_output[idx], save_path + str(idx) + '_threhsold_pred.jpg', threshold=210, size=None)
                        visualize(pred_TAP[idx, 1], save_path + str(idx) + '_TAP.jpg', size=size)

                print('image saved...!! %s %d' % (dir, save_cnt))

        print('testing ended!')
        eval_mean_TAP = sum(test_loss_TAPs) / len(test_loss_TAPs)
        eval_mean_edge = sum(test_loss_edges) / len(test_loss_edges)
        eval_mean_dice = sum(test_loss_dices) / len(test_loss_dices)
        eval_AP = sum(test_APs) / len(test_APs)
        print("\n[TEST] Episode: %d, loss_TAP: %f, loss_edge: %f, loss_dice: %f, AP: %f" % (
            epoch, eval_mean_TAP, eval_mean_edge, eval_mean_dice, eval_AP))

