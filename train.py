#  Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv
import torch.optim as optim
import copy
#Local imports
from data import dataset_s as dset
from models.common import Evaluator
from utils.utils import save_args, load_args
from utils.config_model import configure_model
from params import parser, DATA_FOLDER
import numpy as np
import random

best_auc = 0
best_hm = 0
best_obj = 0
best_attr = 0
best_obj_acc = 0
best_attr_acc = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Get arguments and start logging
    args = parser.parse_args()
    load_args(args.config, args)
    args.name = args.name 
    print(args.name)

    logpath = os.path.join(args.cv_dir, args.name)
    os.makedirs(logpath, exist_ok=True)
    save_args(args, logpath, args.config)
    writer = SummaryWriter(log_dir=logpath, flush_secs=30)

    # Get dataset
    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        split=args.splitname,
        model =args.image_extractor,
        num_negs=args.num_negs,
        pair_dropout=args.pair_dropout,
        update_features = args.update_features,
        train_only= args.train_only,
        open_world=args.open_world
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)
    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER, args.data_dir),
        phase=args.test_set,
        split=args.splitname,
        model=args.image_extractor,
        subset=args.subset,
        update_features=args.update_features,
        open_world=args.open_world
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Get model and optimizer
    image_extractor, model, optimizer, optimizer_ft = configure_model(args, trainset)
    args.extractor = image_extractor

    evaluator_val = Evaluator(testset, model)

    print(model)

    start_obj_epoch = 0
    start_attr_epoch = 0
    start_normal_epoch = 0
    # Load checkpoint
    if args.load is not None:
        checkpoint = torch.load(args.load)
        if image_extractor:
            try:
                image_extractor.load_state_dict(checkpoint['image_extractor'])
                if args.freeze_features:
                    print('Freezing image extractor')
                    image_extractor.eval()
                    for param in image_extractor.parameters():
                        param.requires_grad = False
            except:
                print('No Image extractor in checkpoint')
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)

    best_obj = 0
    for epoch in tqdm(range(start_obj_epoch, args.step_obj + 1), desc = 'Current epoch'):
        train_normal(epoch, image_extractor, model, trainloader, optimizer, writer, step='obj')
        if epoch % args.eval_val_every == 0:
            with torch.no_grad():  # todo: might not be needed
                obj_acc = test_obj(epoch, image_extractor, model, testloader, evaluator_val, writer, args, logpath,
                              step='test')
                if obj_acc > best_obj:
                    best_obj = obj_acc
                    patience = 0
                else:
                    patience += 1
                    if patience > 10:
                        args.load = ospj(logpath, 'ckpt_best_obj_acc.t7')
                        checkpoint = torch.load(args.load)
                        if image_extractor:
                            try:
                                image_extractor.load_state_dict(checkpoint['image_extractor'])
                                image_extractor.eval()
                            except:
                                print('No Image extractor in checkpoint')
                        model.load_state_dict(checkpoint['net'])
                        break
    print('Best best_obj achieved is ', best_obj)

    best_attr = 0
    for epoch in tqdm(range(start_attr_epoch, args.step_attr + 1), desc = 'Current epoch'):
        train_normal(epoch, image_extractor, model, trainloader, optimizer, writer, step='attr')
        if epoch % args.eval_val_every == 0 and epoch > 0:
            with torch.no_grad():  # todo: might not be needed
                attr_acc = test_attr(epoch, image_extractor, model, testloader, evaluator_val, writer, args, logpath,
                         step='test')
                if attr_acc > best_attr:
                    best_attr = attr_acc
                    patience = 0
                else:
                    patience += 1
                    if patience > 15:
                        args.load = ospj(logpath, 'ckpt_best_attr_acc.t7')
                        checkpoint = torch.load(args.load)
                        if image_extractor:
                            try:
                                image_extractor.load_state_dict(checkpoint['image_extractor'])
                                image_extractor.eval()
                            except:
                                print('No Image extractor in checkpoint')
                        model.load_state_dict(checkpoint['net'])
                        break


    print('Best attr_acc achieved is ', best_attr)

    for epoch in tqdm(range(start_normal_epoch, args.step_normal + 1), desc='Current epoch'):
        train_normal(epoch, image_extractor, model, trainloader, optimizer_ft, writer, step='normal')
        if epoch % args.eval_val_every == 0 and epoch > 0:
            with torch.no_grad():  # todo: might not be needed
                if args.fast:
                    test_fast(epoch, image_extractor, model, testloader, evaluator_val, writer, args, logpath, step='test')
    print('Best AUC achieved is ', best_auc)
    print('Best HM achieved is ', best_hm)


def train_normal(epoch, image_extractor, model, trainloader, optimizer, writer, step):
    '''
    train for an epoch
    '''

    if image_extractor:
        image_extractor.train()
    model.train()

    train_loss = 0.0
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc='Training'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0], data[1] = image_extractor(data[0])

        loss, _ = model(data, step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(trainloader)
    writer.add_scalar('Loss/train_total', train_loss, epoch)
    print('{} Epoch: {}| Loss: {}'.format(step, epoch, round(train_loss, 2)))



def test_fast(epoch, image_extractor, model, testloader, evaluator, writer, args, logpath, step):
    '''
    test for an epoch
    '''
    global best_auc, best_hm, best_unseen

    def save_checkpoint(filename, key='AUC'):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            key: stats[key]
        }
        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))

    if image_extractor:
        image_extractor.eval()

    model.eval()


    bias = args.bias
    biaslist = None

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Computing bias'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0], data[1] = image_extractor(data[0])

        scores, _, _, _ = model(data, step)
        scores = scores.to('cpu')

        attr_truth, obj_truth, pair_truth = data[2].to('cpu'), data[3].to('cpu'), data[4].to('cpu')

        biaslist = evaluator.compute_biases(scores.to('cpu'), attr_truth, obj_truth, pair_truth, previous_list = biaslist, bias=bias)

    biaslist = list(evaluator.get_biases(biaslist).numpy())
    biaslist.append(bias)
    if args.partial==True:
        biaslist=[0.0]

    results = {b: {'unseen':0.,'seen':0.,'total_unseen':0.,'total_seen':0., 'attr_match':0.,'obj_match':0.} for b in biaslist}

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0], data[1] = image_extractor(data[0])

        scores, _, _, _ = model(data, step)
        scores = scores.to('cpu')

        attr_truth, obj_truth, pair_truth = data[2].to('cpu'), data[3].to('cpu'), data[4].to('cpu')

        seen_mask = None

        for b in biaslist:
            attr_match, obj_match, seen_match, unseen_match, seen_mask = \
                evaluator.get_accuracies_fast(scores, attr_truth, obj_truth, pair_truth, bias=b, seen_mask=seen_mask)

            results[b]['unseen'] += unseen_match.item()
            results[b]['seen'] += seen_match.item()
            results[b]['total_unseen'] += scores.shape[0]-seen_mask.sum().item()
            results[b]['total_seen'] += seen_mask.sum().item()
            results[b]['attr_match'] += attr_match.item()
            results[b]['obj_match'] += obj_match.item()

    for b in biaslist:
        results[b]['unseen']/= results[b]['total_unseen']
        results[b]['seen']/= results[b]['total_seen']
        results[b]['attr_match']/= (results[b]['total_seen']+results[b]['total_unseen'])
        results[b]['obj_match']/= (results[b]['total_seen']+results[b]['total_unseen'])


    results['a_epoch'] = epoch

    stats = evaluator.collect_results(biaslist,results)

    result = ''
    # write to Tensorboard
    for key in stats:
        writer.add_scalar(key, stats[key], epoch)
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name
    print(f'Test Epoch: {epoch}')
    print(result)

    if stats['AUC'] > best_auc:
            best_auc = stats['AUC']
            print('New best AUC ', best_auc)
            save_checkpoint('best_auc')

    if stats['best_hm'] > best_hm:
            best_hm = stats['best_hm']
            print('New best HM ', best_hm)
            save_checkpoint('best_hm')

    # Logs
    with open(ospj(logpath, 'logs.csv'), 'a') as f:
        w = csv.DictWriter(f, stats.keys())
        if epoch == 0:
            w.writeheader()
        w.writerow(stats)

def test_obj(epoch, image_extractor, model, testloader, evaluator, writer, args, logpath, step):
    '''
    test_obj for an epoch
    '''
    global best_obj_acc

    def save_checkpoint(filename, key='test_obj'):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            # key: stats[key]
        }
        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))


    if image_extractor:
        image_extractor.eval()

    model.eval()
    preds = np.array([])
    targets = np.array([])
    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='test_obj'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0], data[1] = image_extractor(data[0])

        _, _, _, obj_pred = model(data, step)
        attr_truth, obj_truth, pair_truth = data[2], data[3], data[4]

        _, pred = obj_pred.max(1)
        targets = np.append(targets, obj_truth.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    preds = torch.from_numpy(preds)
    targets = torch.from_numpy(targets)
    correct = preds.eq(targets).float().sum(0)
    obj_acc = float(correct / targets.size(0))

    if obj_acc > best_obj_acc:
        best_obj_acc = obj_acc
        save_checkpoint('best_obj_acc')
        print('New best best_obj_acc ', best_obj_acc)

    return obj_acc


def test_attr(epoch, image_extractor, model, testloader, evaluator, writer, args, logpath, step):
    '''
    test_attr for an epoch
    '''
    global best_attr_acc

    def save_checkpoint(filename, key='test_obj'):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            # key: stats[key]
        }
        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        torch.save(state, os.path.join(logpath, 'ckpt_{}.t7'.format(filename)))

    if image_extractor:
        image_extractor.eval()

    model.eval()
    preds = np.array([])
    targets = np.array([])

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='test_attr'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0], data[1] = image_extractor(data[0])

        _, _, attr_pred, _= model(data, step)
        attr_truth, obj_truth, pair_truth = data[2], data[3], data[4]

        _, pred = attr_pred.max(1)
        targets = np.append(targets, attr_truth.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    preds = torch.from_numpy(preds)
    targets = torch.from_numpy(targets)
    correct = preds.eq(targets).float().sum(0)
    attr_acc = float(correct / targets.size(0))

    if attr_acc > best_attr_acc:
        best_attr_acc = attr_acc
        print('New best best_attr_acc ', best_attr_acc)
        save_checkpoint('best_attr_acc')

    return attr_acc

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Best AUC achieved is ', best_auc)
        print('Best HM achieved is ', best_hm)
