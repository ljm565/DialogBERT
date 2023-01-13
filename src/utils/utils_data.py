import torch
from torch.utils.data import Dataset
import random



class DLoader(Dataset):
    def __init__(self, data, tokenizer, config, phase):
        random.seed(999)
        self.data = data
        self.tokenizer = tokenizer
        self.phase = phase
        
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.msk_token_id = self.tokenizer.msk_token_id
        self.pos_pad_token_id = self.tokenizer.pos_pad_token_id
        
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.msk_token = self.tokenizer.msk_token

        self.max_multiturn = config.max_multiturn
        self.max_utterance = config.max_utterance
        self.length = len(self.data)  


    def select_dialog(self, data_list):
        dialog_len = len(data_list)
        st = random.randrange(0, dialog_len - self.cur_turn_len)
        tr = st + self.cur_turn_len
        return data_list[st:tr], data_list[tr]

        
    def add_special_token(self, s, id):
        s = [self.cls_token_id] + self.tokenizer.encode(s)[:self.max_utterance-2] + [self.sep_token_id]
        s += [id] * (self.max_utterance - len(s))
        return s    


    def make_tensor(self, data, id, dtype, src=True):
        if src:
            total_s = []
            for s in data:
                s = self.add_special_token(s, id)
                total_s.append(s)
            pad = [[id] * self.max_utterance] * (self.max_multiturn - len(total_s))
            total_s += pad
            return torch.tensor(total_s, dtype=dtype)
        return torch.tensor(self.add_special_token(data, id), dtype=dtype)


    def make_DUOR_data(self, src_list):
        pair = [(id, s) for id, s in enumerate(src_list)]
        random.shuffle(pair)
        pos_id = [0] + [d[0] + 1 for d in pair] + [len(pair) + 1]
        shuffled_src = [self.cls_token] + [d[1] for d in pair] + [self.sep_token]
        return shuffled_src, pos_id


    def make_MUR_data(self, src_list, idx):
        masking_loc = random.randrange(0, self.cur_turn_len)
        label = [self.cls_token] + src_list + [self.sep_token]

        # to prevent masking short dialogues 
        mlm_probs = [0.0, 0.1, 0.4, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]
        try:
            mlm_prob = mlm_probs[self.cur_turn_len - 1]
        except IndexError:
            mlm_prob = 1
        prob = random.random()

        if prob < mlm_prob:
            prob /= mlm_prob
            # replace one utterance to [CLS] [MSK] [SEP]
            if prob < 0.8:
                src_list[masking_loc] = self.msk_token
                
            # replace other utterance 
            elif prob < 0.9:
                src_list[masking_loc] = self.get_new_s(idx)

        src_list = [self.cls_token] + src_list + [self.sep_token]    

        return src_list, label
            

    def get_new_s(self, idx):
        while 1:
            new_idx = random.randrange(len(self.data))
            if new_idx != idx:
                return random.choice(self.data[new_idx])
        

    def __getitem__(self, idx):
        # define current data length
        self.cur_turn_len = len(self.data[idx]) - 1 if len(self.data[idx]) <= self.max_multiturn - 2 else self.max_multiturn - 2

        """
        Select target dialogues and corresponding target sentence
        src_list: [s1, s2, s3, ..., s{n-1}]
        trg: sn
        """
        src_list, trg = self.select_dialog(self.data[idx])
        
        """
        Data for target (the last utterance for NUG) and vanilla source
        """
        src_nug = [self.cls_token] + list(src_list) + [self.sep_token]
        trg_nug = self.make_tensor(trg, self.pad_token_id, torch.long, False)
        lm_trg = torch.LongTensor([[self.pad_token_id] * self.max_utterance] * self.max_multiturn)
        pos_id = list(range(len(src_list) + 2))
        shuffled = 0

        """
        if
            Data for shuffled DUOR
        else
            Data for MUR
        """
        if self.phase == 'train':
            if random.random() < 0.4 and len(src_list) > 0:
                src_nug, pos_id = self.make_DUOR_data(list(src_list))
                shuffled = 1
            else:
                if len(src_list) > 2:
                    src_nug, lm_trg = self.make_MUR_data(list(src_list), idx)
                    lm_trg = self.make_tensor(lm_trg, self.pad_token_id, torch.long)
                    pos_id = list(range(len(src_list) + 2))
                
        src_nug = self.make_tensor(src_nug, self.pad_token_id, torch.long)
        pos_id = torch.LongTensor(pos_id + [self.pos_pad_token_id] * (self.max_multiturn - len(pos_id)))

        return src_nug, trg_nug, lm_trg, pos_id, shuffled


    def __len__(self):
        return self.length
