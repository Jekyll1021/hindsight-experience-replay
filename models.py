import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params, input_num, output_num=4, ee_pose=False):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(input_num, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, output_num)
        self.ee_pose = ee_pose

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.ee_pose:
            actions = torch.clamp(self.max_action * self.action_out(x), -self.max_action, self.max_action)
        else:
            actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

class critic(nn.Module):
    def __init__(self, env_params, input_num):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(input_num, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value

# define the actor network
class actor_image(nn.Module):
    def __init__(self, env_params, input_num, output_num=4):
        super(actor_image, self).__init__()
        print("creating actor model with image input...")
        self.two_cam = env_params['two_cam']

        # self.feature_extraction_model = models.alexnet(pretrained=True).features.eval()
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.image_fc1 = nn.Linear(256 * 6 * 6, 4096)
        # self.image_fc2 = nn.Linear(4096, 4096)
        # self.image_fc3 = nn.Linear(4096, 64)

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.eval()
        num_ftrs = self.resnet.fc.in_features * 7 * 7
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.resnet.fc = nn.Linear(num_ftrs, 512)

        if self.two_cam:
            self.image_process = nn.Linear(512*2, 512)

        self.input_process = nn.Linear(input_num, 512)

        self.fc1 = nn.Linear(512*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.action_out = nn.Linear(64, output_num)

        self.max_action = env_params['action_max']
        self.depth = env_params['depth']

    def forward(self, x, image):
        if not self.depth:
            image = image.permute((0, 3, 1, 2)).float()
            if self.two_cam:
                image, image2 = torch.split(image, 3, dim=1)
        # image = self.feature_extraction_model(image)
        # image = self.avgpool(image)
        # image = image.view(image.size(0), -1)
        # image = F.relu(self.image_fc1(image))
        # image = F.relu(self.image_fc2(image))
        # image = self.image_fc3(image)

        image = F.relu(self.resnet(image))
        if self.two_cam:
            image2 = F.relu(self.resnet(image2))
            image = torch.cat([image, image2], dim=1)
            image = F.relu(self.image_process(image))

        x = F.relu(self.input_process(x))

        x = torch.cat([x, image], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        print(self.action_out(x))

        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

class critic_image(nn.Module):
    def __init__(self, env_params, input_num):
        super(critic_image, self).__init__()
        self.max_action = env_params['action_max']
        self.two_cam = env_params['two_cam']
        print("creating critic model with image input...")
        # self.feature_extraction_model = models.alexnet(pretrained=True).features.eval()
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.image_fc1 = nn.Linear(256 * 7 * 7, 4096)
        # self.image_fc2 = nn.Linear(4096, 4096)
        # self.image_fc3 = nn.Linear(4096, 64)
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.eval()
        num_ftrs = self.resnet.fc.in_features * 7 * 7
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.resnet.fc = nn.Linear(num_ftrs, 512)

        if self.two_cam:
            self.image_process = nn.Linear(512*2, 512)

        self.input_process = nn.Linear(input_num, 512)

        self.fc1 = nn.Linear(512*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.q_out = nn.Linear(64, 1)
        self.depth = env_params['depth']

    def forward(self, x, image, actions):
        if not self.depth:
            image = image.permute((0, 3, 1, 2)).float()
            if self.two_cam:
                image, image2 = torch.split(image, 3, dim=1)
        # image = self.feature_extraction_model(image)
        # image = self.avgpool(image)
        # image = image.view(image.size(0), -1)
        # image = F.relu(self.image_fc1(image))
        # image = F.relu(self.image_fc2(image))
        # image = self.image_fc3(image)
        image = F.relu(self.resnet(image))
        if self.two_cam:
            image2 = F.relu(self.resnet(image2))
            image = torch.cat([image, image2], dim=1)
            image = F.relu(self.image_process(image))

        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.input_process(x))

        x = torch.cat([x, image], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = F.sigmoid(self.q_out(x))

        return q_value

class actor_recurrent(nn.Module):
    def __init__(self, env_params, input_num, output_num=4, ee_pose=False):
        super(actor_recurrent, self).__init__()
        self.max_action = env_params['action_max']
        self.hidden_size = 64
        self.input_num = input_num
        self.gru = nn.GRU(input_num, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, output_num)
        self.ee_pose = ee_pose

    def _forward_gru(self, x, hxs):
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        x = x.view(1, -1, self.input_num)
        hxs = hxs.view(1, -1, self.hidden_size)
        x, hxs = self.gru(
            x,
            hxs
        )

        return x.squeeze(0), hxs.squeeze(0)

    def forward(self, x, hidden):
        x, hidden = self._forward_gru(x, hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.ee_pose:
            actions = torch.clamp(self.max_action * self.action_out(x), -self.max_action, self.max_action)
        else:
            actions = self.max_action * torch.tanh(self.action_out(x))

        return actions, hidden

class critic_recurrent(nn.Module):
    def __init__(self, env_params, input_num):
        super(critic_recurrent, self).__init__()
        self.max_action = env_params['action_max']
        self.hidden_size = 64
        self.input_num = input_num
        self.gru = nn.GRU(input_num, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def _forward_gru(self, x, hxs):
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        x = x.view(1, -1, self.input_num)
        hxs = hxs.view(1, -1, self.hidden_size)
        x, hxs = self.gru(
            x,
            hxs
        )

        return x.squeeze(0), hxs.squeeze(0)

    def forward(self, x, actions, hidden):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x, hidden = self._forward_gru(x, hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value, hidden

class actor_image_recurrent(nn.Module):
    def __init__(self, env_params, input_num, output_num=4, ee_pose=False):
        super(actor_image_recurrent, self).__init__()
        self.max_action = env_params['action_max']
        self.hidden_size = 64
        self.input_num = input_num

        self.feature_extraction_model = models.alexnet(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.image_fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.image_fc2 = nn.Linear(4096, 4096)
        self.image_fc3 = nn.Linear(4096, 64)

        self.gru = nn.GRU(input_num + 64, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, output_num)
        self.ee_pose = ee_pose

    def _forward_gru(self, x, hxs):
        # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
        x = x.view(1, -1, self.input_num)
        hxs = hxs.view(1, -1, self.hidden_size)
        x, hxs = self.gru(
            x,
            hxs
        )

        return x.squeeze(0), hxs.squeeze(0)

    def forward(self, x, image, hidden):
        image = image.permute((0, 3, 1, 2)).float()
        img = self.feature_extraction_model.features(image)
        img = self.avgpool(img)
        img = img.view(img.size(0), -1)
        img = F.relu(self.image_fc1(img))
        img = F.relu(self.image_fc2(img))
        img = self.image_fc3(img)

        x = torch.cat([x, img], dim=1)

        x, hidden = self._forward_gru(x, hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.ee_pose:
            actions = torch.clamp(self.max_action * self.action_out(x), -self.max_action, self.max_action)
        else:
            actions = self.max_action * torch.tanh(self.action_out(x))

        return actions, hidden

class open_loop_image_predictor(nn.Module):
    def __init__(self, input_num, output_num=1, hidden=512):
        super(open_loop_image_predictor, self).__init__()
        self.input_num = input_num

        self.feature_extraction_model = models.alexnet(pretrained=True)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.image_fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.image_fc2 = nn.Linear(4096, 1024)
        self.image_fc3 = nn.Linear(1024, hidden)

        self.input_fc1 = nn.Linear(input_num, hidden)
        self.input_fc2 = nn.Linear(hidden, hidden)

        self.fc1 = nn.Linear(hidden*2, hidden)
        self.q = nn.Linear(hidden, output_num)

    def forward(self, input, image):
        image = image.permute((0, 3, 1, 2)).float()
        img = self.feature_extraction_model.features(image)
        img = self.avgpool(img)
        img = img.view(img.size(0), -1)
        img = F.relu(self.image_fc1(img))
        img = F.relu(self.image_fc2(img))
        img = self.image_fc3(img)

        input = self.input_fc1(input)
        input = self.input_fc2(input)

        x = torch.cat([input, img], dim=1)

        x = F.relu(self.fc1(x))
        score = torch.sigmoid(self.q(x))

        return score
