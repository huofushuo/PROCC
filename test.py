#  Torch imports
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import numpy as np

cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import pickle
import pickle
# Local imports
from data import dataset_s as dset
from models.common import Evaluator
from utils.utils import load_args
from utils.config_model import configure_model
from params import parser, DATA_FOLDER

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():

    # Get arguments and start logging
    args = parser.parse_args()
    logpath = args.logpath
    config = [os.path.join(logpath, _) for _ in os.listdir(logpath) if _.endswith('yml')][0]
    load_args(config, args)
    if args.open_world:
        print('Testing in an open world')

    # Get dataset
    trainset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='train',
        split=args.splitname,
        model=args.image_extractor,
        update_features=args.update_features,
        train_only=args.train_only,
        subset=args.subset,
        open_world=args.open_world
    )

    valset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='val',
        split=args.splitname,
        model=args.image_extractor,
        subset=args.subset,
        update_features=args.update_features,
        open_world=args.open_world
    )

    valoader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8)

    testset = dset.CompositionDataset(
        root=os.path.join(DATA_FOLDER,args.data_dir),
        phase='test',
        split=args.splitname,
        model =args.image_extractor,
        subset=args.subset,
        update_features = args.update_features,
        open_world=args.open_world
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers)


    # Get model and optimizer
    image_extractor, model, optimizer = configure_model(args, trainset)
    args.extractor = image_extractor


    args.load = ospj(logpath,'ckpt_best_hm.t7')

    checkpoint = torch.load(args.load)
    if image_extractor:
        try:
            image_extractor.load_state_dict(checkpoint['image_extractor'])
            image_extractor.eval()
        except:
            print('No Image extractor in checkpoint')

    # print(checkpoint['net'])
    for k, v in checkpoint['net'].items():
        print(k)
    print(model)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    threshold = None
    if args.open_world and args.hard_masking:
        assert args.model == 'kgsp' or args.model == 'compcos' or args.model == 'visprodb' or args.model == 'graphfull', args.model + ' does not have hard masking.'
        if args.threshold is not None:
            threshold = float(args.threshold)
            model.compute_feasibility()
        else:
            evaluator_val = Evaluator(valset, model)
            unseen_scores = model.compute_feasibility().to('cpu')
            seen_mask = model.seen_mask.to('cpu')
            min_feasibility = (unseen_scores+seen_mask*10.).min()
            max_feasibility = (unseen_scores-seen_mask*10.).max()
            thresholds = np.linspace(min_feasibility,max_feasibility, num=args.threshold_trials)
            best_auc = 0.
            best_th = -10
            with torch.no_grad():
                for th in thresholds:
                    results = test_fast(image_extractor,model,valoader,evaluator_val,args,threshold=th,print_results=False)
                    #print(results)
                    auc = results['AUC']
                    if auc > best_auc:
                        best_auc = auc
                        best_th = th
                        print('New best AUC',best_auc)
                        print('Threshold',best_th)

            threshold = best_th

    evaluator = Evaluator(testset, model)

    with torch.no_grad():
        if args.fast:
            test_fast(image_extractor, model, testloader, evaluator, args, threshold=threshold) #TODO hard masking


def test_fast(image_extractor, model, testloader, evaluator,  args, print_results=True,hard_mask=None, threshold=None,biaslist=None):
    '''
    Runs testing for an epoch
    '''


    if image_extractor:
        image_extractor.eval()

    model.eval()


    bias = args.bias
    biaslist = None
    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Computing bias'):

        data = [d.to(device) for d in data]

        if image_extractor:
            data[0], data[1] = image_extractor(data[0])

        scores, _, _, _ = model(data, step = 0)
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

        if threshold is not None:
            scores,_ = model.val_forward_with_threshold(data,threshold)
        else:
            scores, _, _, _ = model(data, step = 0)
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

    stats = evaluator.collect_results(biaslist,results)

    result = ''
    # write to Tensorboard
    for key in stats:
        result = result + key + '  ' + str(round(stats[key], 5)) + '| '

    result = result + args.name

    if print_results:
        print(f'Results')
        print(result)
    return stats, biaslist


if __name__ == '__main__':
    main()
