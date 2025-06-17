from __future__ import print_function
import argparse, sys, time, torch
import torch
print(torch.cuda.is_available())  # Should return True if CUDA is properly set up
print(torch.cuda.device_count())  # Should return the number of available GPUs
print(torch.version.cuda)  
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from utils import *
from data_manager import *
from data_loader import SYSUData, RegDBData, Dataloader_MEM, TestData, ChannelExchange, LLCMData
from memory import ClusterMemory
from model import Model
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from tensorboardX import SummaryWriter
from distillModel import DistillModel 

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--phase', default='debug', type=str, help='debug or train')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.00035 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='adam', type=str, help='optimizer')
parser.add_argument('--resume', '-r', default='', type=str,help='resume from checkpoint')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--rerank', default= "no" , type=str, metavar='rerank', help='gamma for the hard mining')
parser.add_argument('--num_pos', default=4, type=int,help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--shot', default=1, type=int, help='select single-shot or multi-shot')
parser.add_argument('--channel', default= 2 , type=int, metavar='channel', help='gamma for the hard mining')      
parser.add_argument('--ml', default= 1 , type=int, metavar='ml', help='gamma for the hard mining') 
parser.add_argument('--kl', default= 1.2 , type=float, metavar='kl', help='use kl loss and the weight')
parser.add_argument('--sid', default= 1 , type=float, metavar='kl', help='use kl loss and the weight')  
parser.add_argument('--mem', default= 1 , type=float, metavar='mem', help='memory')
parser.add_argument('--nhard', action='store_false', help='not use hard memory')
parser.add_argument('--nce', action='store_true', help='not use cross entropy loss')
parser.add_argument('--gc', default=1, type=float, help='global center loss')
parser.add_argument('--caid', default= 0.4 , type=float, metavar='id loss for ca', help='use kl loss and the weight')
parser.add_argument('--mem_up', default= 0.25 , type=float, metavar='id loss for ca', help='use kl loss and the weight')
parser.add_argument('--gx', default= 0 , type=int, help='if use updated memory')
parser.add_argument('--ablation', type=str, default='full_mlkd', 
                    choices=['only_kd', 'mlkd_no_instance', 'mlkd_no_class', 'full_mlkd'],
                    help='which ablation configuration to run')
# Path arguments - you need to specify your own paths
parser.add_argument('--data_path', default='YOUR_DATA_PATH', type=str, help='path to dataset directory')
parser.add_argument('--model_path', default='YOUR_MODEL_PATH', type=str, help='model save path')
parser.add_argument('--log_path', default='YOUR_LOG_PATH', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='YOUR_VIS_LOG_PATH', type=str, help='tensorboard log save path')
parser.add_argument('--teacher_path', type=str, default='YOUR_TEACHER_MODEL_PATH', help='path to teacher model')


args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(0)
cudnn.deterministic = True
cudnn.benchmark = False
dataset = args.dataset
if dataset == 'sysu':
    data_path = args.data_path
    log_path = args.log_path 
    test_mode = [1, 2]  # thermal to visible
    args.img_w, args.img_h = 128, 384
elif dataset == 'regdb':
    data_path = args.data_path
    log_path = args.log_path 
    test_mode = [2, 1]  # visible to thermal-[2,1]
    args.img_w, args.img_h = 144, 288 
    args.num_pos = 5
checkpoint_path = args.model_path
if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)
suffix = args.phase+'_' + dataset
suffix += '_mem' if args.mem!=0 else ''
suffix = suffix + '_p{}_n{}_ch{}_lr_{}'.format(args.num_pos, args.batch_size,(args.channel), args.lr) 
if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim
if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)
sys.stdout = Logger(log_path + suffix + '.txt')
vis_log_dir = args.vis_log_path + suffix + '/'
if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print('========')
arg_s = "Args:{}".format(args)
for i in range(len(arg_s)//130+1):
    print(arg_s[0+130*i:130+130*i])
print('========')


best_acc = 0  # best test accuracy
start_epoch = 0
print('==> Loading data..')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_center = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
    ChannelExchange(gray = 2)]) 
transform_test = transforms.Compose( [
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])
end = time.time()
if dataset == 'sysu':
    trainset = SYSUData(data_path, transform=None,size=(args.img_h,args.img_w))
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)

    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0, shot=args.shot)
elif dataset == 'regdb':
    trainset = RegDBData(data_path, args.trial, transform=None,size=(args.img_w, args.img_h))
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal=test_mode[0])
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal=test_mode[1])
gallset  = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)
print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))
print('==> Building model..')

n_class = len(np.unique(trainset.train_color_label))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(2)

# Load single teacher model
teacher_model = Model(n_class, arch='resnet50',
                     mutual_information=(args.ml!=0),
                     drop_last_stride=True,  
                     weight_KL=args.kl,
                     weight_sid=args.sid, 
                     classification=args.nce,
                     channel = args.channel,
                     mem = args.mem,
                     global_center =args.gc,
                     weight_caid=args.caid,
                     gx = args.gx
                     )
checkpoint = torch.load(args.teacher_path)
teacher_model.load_state_dict(checkpoint['net'], strict=False)
teacher_model.to(device)

student_model =  Model(n_class, arch='resnet50',
            mutual_information=(args.ml!=0),
            drop_last_stride=True,  
            weight_KL=args.kl,
            weight_sid=args.sid, 
            classification=args.nce,
            channel = args.channel,
            mem = args.mem,
            global_center =args.gc,
            weight_caid=args.caid,
            gx = args.gx
    )
student_model.to(device)

# Initialize the distillation model
distillation_model = DistillModel(
    teacher=teacher_model,
    student=student_model,
)
distillation_model.to(device)


# Define optimizer for student model
optimizer = optim.Adam(distillation_model.student.parameters(), lr=args.lr)

def flooded_loss(original_loss, b: float):
    """
    Flooding Loss as described in Ishida et al. (2020):
    Flooded Loss = |original_loss - b| + b
    """
    return torch.abs(original_loss - b) + b


def get_ablation_config(name):
    """Get model configuration for specific ablation"""
    configs = {
        'only_kd': {
            'name': 'Only KD',
            'mlkd': True,
            'kd': True,
            'instance': False,
            'class': False
        },
        'mlkd_no_instance': {
            'name': 'MLKD without Instance Loss',
            'mlkd': True,
            'kd': False,
            'instance': False,
            'class': True
        },
        'mlkd_no_class': {
            'name': 'MLKD without Class Loss',
            'mlkd': True,
            'kd': False,
            'instance': True,
            'class': False
        },
        'full_mlkd': {
            'name': 'Full MLKD',
            'mlkd': True,
            'kd': False,
            'instance': True,
            'class': True
        }
    }
    return configs[name]

def configure_model_for_ablation(model, config):
    """Configure model based on ablation settings"""
    model.mlkd = config['mlkd']
    model.use_kd = config['kd']
    model.use_instance = config['instance']
    model.use_class = config['class']
    
    
    model.mlkd_loss.use_kd_loss = config['kd']
    model.mlkd_loss.use_instance_loss = config['instance']
    model.mlkd_loss.use_class_loss = config['class']
    
    return model

def train_distillation(epoch, trainloader, config):
    """Training with specific ablation configuration"""
    distillation_model.train()
    
    for batch_idx, (rgb_inputs, aug_inputs, ir_inputs, rgb_labels, ir_labels) in enumerate(trainloader):
        rgb_inputs, ir_inputs = rgb_inputs.to(device), ir_inputs.to(device)
        rgb_labels = rgb_labels.to(device)

        losses_dict = distillation_model(rgb_inputs, ir_inputs, rgb_labels)
        
        # 1) Forward pass
        losses_dict = distillation_model(rgb_inputs, ir_inputs, rgb_labels)

        # 2) Get original total loss
        total_loss = losses_dict["total_loss"]

        # 3) Backprop & update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Log losses with configuration prefix
        suffix = f"/{args.ablation}"
        for key, value in losses_dict.items():
            if isinstance(value, torch.Tensor):
                writer.add_scalar(f'Loss{suffix}/{key}', value.item(), 
                                epoch * len(trainloader) + batch_idx)
        

        if batch_idx % 50 == 0:
            loss_str = [f'Config: {config["name"]}',
                       f'Epoch: {epoch}',
                       f'Batch: {batch_idx}/{len(trainloader)}',
                       f'Total Loss: {losses_dict["total_loss"]}']

            
            if config['kd']:
                loss_str.append(f'KD Loss: {losses_dict.get("loss_kd", 0)}')
                loss_str.append(f'KD Strong Loss: {losses_dict.get("loss_kd_strong", 0)}')
                loss_str.append(f'KD Weak Loss: {losses_dict.get("loss_kd_weak", 0)}')

            
            if config['mlkd']:
                if config['instance']:
                    loss_str.extend([
                        f'Instance Loss: {losses_dict.get("loss_instance", 0)}',
                        f'Instance Strong: {losses_dict.get("loss_instance_strong", 0)}',
                        f'Instance Weak: {losses_dict.get("loss_instance_weak", 0)}'
                    ])
                if config['class']:
                    loss_str.extend([
                        f'Class Loss: {losses_dict.get("loss_class", 0)}',
                        f'Class Strong: {losses_dict.get("loss_class_strong", 0)}',
                        f'Class Weak: {losses_dict.get("loss_class_weak", 0)}'
                    ])
            
            print(' | '.join(loss_str))

    return losses_dict

# Extract gallery features using the student model
def Extract_Gallery():
    student_model.eval()  # Use student model after distillation
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, student_model.out_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = input.to(device)     
            feat = student_model(input, modal=1)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    return gall_feat

# Extract query features using the student model
def Extract_Query():
    student_model.eval()
    ptr = 0
    query_feat = np.zeros((nquery, student_model.out_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = input.to(device)
            feat = student_model(input, modal=2)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    return query_feat

def test(epoch, query_feat=None, print_log=True):
    gall_feat = Extract_Gallery()
    if query_feat is None:
        query_feat = Extract_Query()
    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    if args.rerank=="k":
        distmat = -k_reciprocal(query_feat, gall_feat)
    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

    if print_log:
        writer.add_scalar('rank1', cmc[0], epoch)
        writer.add_scalar('mAP', mAP, epoch)
        writer.add_scalar('mINP', mINP, epoch)
    return cmc, mAP, mINP

def test_student(epoch):
    query_feat = Extract_Query()
    global gall_loader, query_loader
    if dataset == 'sysu':
        for trial in range(10):
            gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial, shot=args.shot)
            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            cmc, mAP, mINP = test(epoch, query_feat=query_feat)
            if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP
            else:
                all_cmc = all_cmc + cmc
                all_mAP = all_mAP + mAP
                all_mINP = all_mINP + mINP

        print('Mean:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                all_cmc[0]/10, all_cmc[4]/10, all_cmc[9]/10, all_cmc[19]/10, all_mAP/10, all_mINP/10))
        return all_cmc/10, all_mAP/10, all_mINP/10
    else:
        cmc, mAP, mINP = test(start_epoch,query_feat=query_feat)
        print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        return cmc, mAP, mINP

def initialize_memory():
    print('    Generate classes center for memory...')
    teacher_model.eval()
    Memset = Dataloader_MEM(data_path, dataset=trainset,size=(args.img_h,args.img_w))
    memory = [ClusterMemory(teacher_model.out_dim,n_class,use_hard=args.nhard,momentum=0.1).to(device)]
    memory.append(ClusterMemory(teacher_model.out_dim,n_class,use_hard=False,momentum=args.mem_up).to(device))
    memory.append(ClusterMemory(teacher_model.out_dim,n_class,use_hard=False,momentum=args.mem_up).to(device))
    # memory.append(ClusterMemory(teacher_model.out_dim,n_class,use_hard=False,momentum=args.mem_up).to(device))

    ''' Global  ||  RGB  ||  IR  ||  Aux '''
    # centers = torch.zeros(args.channel + 1, n_class, teacher_model.out_dim).cuda()
    centers = torch.zeros(args.channel+1, n_class, teacher_model.out_dim).cuda()

    log = torch.zeros(args.channel, n_class).cuda()
    for c in range(args.channel):
        Memset.choose = c 
        memloader = data.DataLoader(Memset, batch_size=args.test_batch,sampler=None, num_workers=args.workers, drop_last=False)
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(memloader):
                unique = label.unique()
                input = Variable(input.cuda())
                feat = teacher_model(input,modal=c+1-(c==2)*1)
                for i in unique:
                    log[c][i] += label.eq(i).sum()
                    centers[c+1][i] += feat[label.eq(i)].sum(0)
            centers[c+1] /= log[c].unsqueeze(1)
            memory[c+1].features = F.normalize(centers[c+1], dim=1)
            memory[c+1].centers = memory[c+1].features.clone()
            memory[0].features += memory[c+1].features
    memory[0].features /= args.channel
    memory[0].centers = memory[0].features.clone()
    print('    Generate OK!!,  Got ', n_class, ' Classes')
    teacher_model.memory, teacher_model.memory_rgb, teacher_model.memory_ir = memory[0], memory[1], memory[2]
        
def run_training():
    """Main training loop with ablation support"""
    global best_mAP, distillation_model
    best_mAP = 0
    best_epoch = 0
    max_epochs = 160
    
    config = get_ablation_config(args.ablation)
    print(f"\nRunning ablation: {config['name']}")
    
    # Configure model once at start
    distillation_model = configure_model_for_ablation(distillation_model, config)
    

    if args.mem!=0:
        initialize_memory()

    for epoch in range(start_epoch, max_epochs):
        sampler = IdentitySampler(trainset.train_color_label,
                                trainset.train_thermal_label,
                                color_pos, thermal_pos,
                                args.num_pos, args.batch_size,
                                epoch)
        trainset.cIndex = sampler.index1
        trainset.tIndex = sampler.index2
        loader_batch = args.batch_size * args.num_pos
        trainloader = data.DataLoader(trainset,
                                    batch_size=loader_batch,
                                    sampler=sampler,
                                    num_workers=args.workers,
                                    drop_last=True)
        
        # Train and test
        train_distillation(epoch, trainloader, config)
        cmc, mAP, mINP = test_student(epoch)

        # Save best model
        if mAP > best_mAP:
            print(f"New best mAP: {mAP:.4f}, saving model...")
            best_mAP = mAP
            best_epoch = epoch
            state = {
                'net': student_model.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
                'ablation': args.ablation,
                'config': config
            }
            save_name = f"{args.model_path}/{args.ablation}_best.pth"
            torch.save(state, save_name)

        # Log results
        writer.add_scalar(f'Metrics/{args.ablation}/rank1', cmc[0], epoch)
        writer.add_scalar(f'Metrics/{args.ablation}/mAP', mAP, epoch)
        writer.add_scalar(f'Metrics/{args.ablation}/mINP', mINP, epoch)

    return best_mAP, best_epoch

def save_ablation_results(mAP, epoch, config):
    """Save ablation results to file"""
    results_file = os.path.join(args.log_path, 'ablation_results.txt')
    with open(results_file, 'a') as f:
        f.write(f"\nAblation: {config['name']}\n")
        f.write(f"Best mAP: {mAP:.2%}\n")
        f.write(f"Best Epoch: {epoch}\n")
        f.write("-" * 50 + "\n")

if __name__ == '__main__':
    # Run the training with selected ablation
    best_mAP, best_epoch = run_training()
    
    # Save and display results
    config = get_ablation_config(args.ablation)
    save_ablation_results(best_mAP, best_epoch, config)
    
    print(f"\nAblation {config['name']} completed:")
    print(f"Best mAP: {best_mAP:.2%}")
    print(f"Best Epoch: {best_epoch}")
    
    writer.close()