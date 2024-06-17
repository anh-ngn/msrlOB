import torch
import torchvision
from models.Q_net import Q_zoom, Q_refine
from data import load_images_names_in_data_set, get_bb_of_gt_from_pascal_xml_annotation
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from PIL import Image, ImageDraw
from utils import cal_iou, reward_func
import time

# hyper-parameters
BATCH_SIZE = 100
LR = 1e-6
GAMMA = 0.9
MEMORY_CAPACITY = 1000
Q_NETWORK_ITERATION = 100
epochs = 50
epochs = 11
NUM_ACTIONS = 6
his_actions = 4
subscale = 3/4
NUM_STATES = 7*7*512+his_actions*NUM_ACTIONS
path_voc = "/home/hanj/dataset/VOCdevkit/VOC2007/"
path_voc = "/kaggle/input/pascal-voc-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/"
path_voc_test = "/kaggle/input/pascal-voc-2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/"


class DQN():
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.eval_net, self.target_net = Q_zoom(), Q_zoom()
        self.eval_net.to(self.device)
        self.target_net.to(self.device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, EPISILO):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        if np.random.randn() <= EPISILO:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].cpu().item()
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(
            batch_memory[:, :NUM_STATES]).to(self.device)
        batch_action = torch.LongTensor(
            batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int)).to(self.device)
        batch_reward = torch.FloatTensor(
            batch_memory[:, NUM_STATES+1:NUM_STATES+2]).to(self.device)
        batch_next_state = torch.FloatTensor(
            batch_memory[:, -NUM_STATES:]).to(self.device)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target_unterminated = batch_reward + GAMMA * \
            q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_target = torch.where(
            batch_action != 5, q_target_unterminated, batch_reward)
        loss = self.loss_func(q_eval, q_target)
        # print("step loss is {:.3f}".format(loss.cpu().detach().item()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def init_process(image, transform=None):
    if transform:
        image = transform(image)
    return image.unsqueeze(0)


def inter_process(image, bbx, transform=None):
    (left, upper, right, lower) = (bbx[0], bbx[2], bbx[1], bbx[3])
    image_crop = image.crop((left, upper, right, lower))
    if transform:
        image_crop = transform(image_crop)
    return image_crop.unsqueeze(0)


def update_bbx(bbx, action):
    new_bbx = np.zeros(4)
    if action == 0:  # top left
        new_bbx[0] = bbx[0]
        new_bbx[1] = bbx[0] + (bbx[1] - bbx[0]) * subscale
        new_bbx[2] = bbx[2]
        new_bbx[3] = bbx[2] + (bbx[3] - bbx[2]) * subscale
    elif action == 1:  # top right
        new_bbx[0] = bbx[1] - (bbx[1] - bbx[0]) * subscale
        new_bbx[1] = bbx[1]
        new_bbx[2] = bbx[2]
        new_bbx[3] = bbx[2] + (bbx[3] - bbx[2]) * subscale
    elif action == 2:  # lower left
        new_bbx[0] = bbx[0]
        new_bbx[1] = bbx[0] + (bbx[1] - bbx[0]) * subscale
        new_bbx[2] = bbx[3] - (bbx[3] - bbx[2]) * subscale
        new_bbx[3] = bbx[3]
    elif action == 3:  # lower right
        new_bbx[0] = bbx[1] - (bbx[1] - bbx[0]) * subscale
        new_bbx[1] = bbx[1]
        new_bbx[2] = bbx[3] - (bbx[3] - bbx[2]) * subscale
        new_bbx[3] = bbx[3]
    elif action == 4:  # center
        new_bbx[0] = (bbx[0] + bbx[1]) / 2 - (bbx[1] - bbx[0]) * subscale / 2
        new_bbx[1] = (bbx[0] + bbx[1]) / 2 + (bbx[1] - bbx[0]) * subscale / 2
        new_bbx[2] = (bbx[2] + bbx[3]) / 2 - (bbx[3] - bbx[2]) * subscale / 2
        new_bbx[3] = (bbx[2] + bbx[3]) / 2 + (bbx[3] - bbx[2]) * subscale / 2
    elif action == 5:
        new_bbx = bbx
    return new_bbx


def draw_bounding_box(image, bbx, epoch, step, image_name):
    draw = ImageDraw.Draw(image)
    draw.rectangle([bbx[0], bbx[2], bbx[1], bbx[3]], outline="red", width=2)
    print(image_name, epoch, step)
    image.save(f"bounding_box_{image_name}_epoch_{epoch}_step_{step}.jpg")


def main(args):
    best_reward = float("-inf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_names = np.array(load_images_names_in_data_set(
        'aeroplane_trainval', path_voc))
    feature_exactrator = torchvision.models.vgg16(
        pretrained=True).features.to(device)
    single_plane_image_names = []
    single_plane_image_gts = []
    dqn = DQN(device)
    EPISILO = args.EPISILO
    subscale = args.Subscale
    for image_name in image_names:
        annotation = get_bb_of_gt_from_pascal_xml_annotation(
            image_name, path_voc)
        if len(annotation) > 1:
            continue
        single_plane_image_names.append(image_name)
        single_plane_image_gts.append(annotation[0][1:])  # [[x1,x2,y1,y2] ...]

    trans = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    for i in range(epochs):
        ep_reward = 0
        for index, image_name in enumerate(single_plane_image_names):
            image_path = os.path.join(
                path_voc + "JPEGImages", image_name + ".jpg")
            image_original = Image.open(image_path)
            width, height = image_original.size
            bbx_gt = single_plane_image_gts[index]

            image = init_process(image_original, trans).to(device)
            bbx = [0, width, 0, height]
            history_action = np.zeros(his_actions * NUM_ACTIONS)
            with torch.no_grad():
                vector = feature_exactrator(image).cpu(
                ).detach().numpy().reshape(7 * 7 * 512)
            state = np.concatenate([history_action, vector])
            step = 0
            while step < 10:
                iou = cal_iou(bbx, bbx_gt)
                if iou > 0.5:
                    action = 5
                else:
                    action = dqn.choose_action(state, EPISILO)
                new_bbx = update_bbx(bbx, action)
                reward = reward_func(bbx, new_bbx, bbx_gt, action)

                action_vec = np.zeros(NUM_ACTIONS)
                action_vec[action] = 1.0
                history_action = np.concatenate(
                    [history_action[NUM_ACTIONS:], action_vec])

                with torch.no_grad():
                    vector = feature_exactrator(inter_process(image_original, new_bbx, trans).to(
                        device)).cpu().detach().numpy().reshape(7 * 7 * 512)
                next_state = np.concatenate([history_action, vector])

                dqn.store_transition(state, action, reward, next_state)

                ep_reward += reward

                if dqn.memory_counter >= MEMORY_CAPACITY:
                    print("episode: {},".format(i), end=' ')
                    dqn.learn()

                if action == 5:
                    break

                state = next_state
                bbx = new_bbx
                step += 1

                # Save bounding box for image 009472 after each 10 epochs
                if image_name == "009472" and (i + 1) % 10 == 0:
                    draw_bounding_box(image_original, bbx,
                                      i + 1, step, image_name)

        if EPISILO > 0.1:
            EPISILO -= 0.05
        print("episode: {} , this epoch reward is {}".format(
            i, round(ep_reward, 3)))

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(dqn.eval_net.state_dict(), 'eval_net.pth')
            print("model saved")

    dqn.eval_net.load_state_dict(torch.load('eval_net.pth'))
    dqn.eval_net.eval()
    dqn.eval_net.to(device)
    print("model loaded")

    image_names = np.array(load_images_names_in_data_set(
        'aeroplane_test', path_voc_test))
    single_plane_image_names = []
    single_plane_image_gts = []
    for image_name in image_names:
        annotation = get_bb_of_gt_from_pascal_xml_annotation(
            image_name, path_voc_test)
        if len(annotation) > 1:
            continue
        single_plane_image_names.append(image_name)
        single_plane_image_gts.append(annotation[0][1:])

    np.save('single_plane_image_names.npy', single_plane_image_names)
    np.save('single_plane_image_gts.npy', single_plane_image_gts)
    print("single_plane_image_names and single_plane_image_gts saved")

    total_iou = 0
    for index, image_name in enumerate(single_plane_image_names):
        image_path = os.path.join(
            path_voc_test + "JPEGImages", image_name + ".jpg")
        image_original = Image.open(image_path)
        width, height = image_original.size
        bbx_gt = single_plane_image_gts[index]

        image = init_process(image_original, trans).to(device)
        bbx = [0, width, 0, height]
        history_action = np.zeros(his_actions * NUM_ACTIONS)
        with torch.no_grad():
            vector = feature_exactrator(image).cpu(
            ).detach().numpy().reshape(7 * 7 * 512)
        state = np.concatenate([history_action, vector])
        step = 0
        while step < 10:
            iou = cal_iou(bbx, bbx_gt)
            if iou > 0.5:
                action = 5
            else:
                action = dqn.choose_action(state, EPISILO)
            new_bbx = update_bbx(bbx, action)
            if action == 5:
                break
            state = np.concatenate([history_action, vector])
            bbx = new_bbx
            step += 1
        total_iou += cal_iou(bbx, bbx_gt)
        with open('bbx.txt', 'a') as f:
            f.write("bbx: {}, bbx_gt: {}\n".format(bbx, bbx_gt))
    print("average IOU is {:.3f}".format(
        total_iou / len(single_plane_image_names)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hierarchical Object Detection with Deep Reinforcement Learning')
    parser.add_argument('--gpu-devices', default='1', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use_gpu', default=True, action='store_true')
    parser.add_argument('--EPISILO', type=int, default=1.0)
    parser.add_argument('--Subscale', type=float, default=3/4)
    parser.add_argument('--image_name', type=str, default='001373',
                        help='name of the image for demonstration')
    args = parser.parse_args()
    main(args)
