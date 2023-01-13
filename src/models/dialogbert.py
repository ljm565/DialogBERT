import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertLMHeadModel, BertConfig, BertForPreTraining



# utterance encoder
class UtteranceEncoder(nn.Module):
    def __init__(self, config):
        super(UtteranceEncoder, self).__init__()
        # self.bert_config = BertConfig.from_pretrained(config.pretrained_model)
        # self.model = BertModel(self.bert_config)
        # self.pretrained_model = config.pretrained_model
        # self.model = BertForPreTraining.from_pretrained(self.pretrained_model)
        self.encoder_config = BertConfig(vocab_size=30522, hidden_size=256, num_hidden_layers=6, num_attention_heads=2, intermediate_size=1024)
        self.model = BertForPreTraining(self.encoder_config)


    def forward(self, x, mask):
        batch_size, max_multiturn, max_utterance = x.size()   # B x multi_turn x max_len
        x, mask = x.view(-1, max_utterance), mask.view(-1, max_utterance)
        output = self.model.bert(input_ids=x, attention_mask=mask)['pooler_output']    # (B x multi_turn) x hidden_dim         
        output = output.view(batch_size, max_multiturn, -1)   # B x multi_turn x hidden_dim
        return output



# context encoder
class ContextEncoder(nn.Module):
    def __init__(self, config):
        super(ContextEncoder, self).__init__()
        # self.bert_config = BertConfig.from_pretrained(config.pretrained_model)
        # self.model = BertModel(self.bert_config)
        # self.pretrained_model = config.pretrained_model
        # self.model = BertModel.from_pretrained(self.pretrained_model)
        self.encoder_config = BertConfig(vocab_size=30522, hidden_size=256, num_hidden_layers=6, num_attention_heads=2, intermediate_size=1024)
        self.model = BertModel(self.encoder_config)


    def forward(self, x, mask):
        output = self.model(inputs_embeds=x, attention_mask=mask)['last_hidden_state']   # B x multi_turn x hidden_dim 
        return output



# NN for MUR
class MaskedUtteranceRegression(nn.Module):
    def __init__(self, config):
        super(MaskedUtteranceRegression, self).__init__()
        # self.bert_config = BertConfig.from_pretrained(config.pretrained_model)
        self.bert_config = BertConfig(vocab_size=30522, hidden_size=256, num_hidden_layers=6, num_attention_heads=2, intermediate_size=1024)
        self.hidden_dim = self.bert_config.hidden_size
        self.dropout = self.bert_config.hidden_dropout_prob
        self.layer_norm_eps = self.bert_config.layer_norm_eps

        self.encoding_converter = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim, eps=self.layer_norm_eps)
        )

    
    def forward(self, x):
        return self.encoding_converter(x)



# NN for DUOR
class DistributedUtteranceOrderRanking(nn.Module):
    def __init__(self, config, tokenizer):
        super(DistributedUtteranceOrderRanking, self).__init__()
        # self.bert_config = BertConfig.from_pretrained(config.pretrained_model)
        self.bert_config = BertConfig(vocab_size=30522, hidden_size=256, num_hidden_layers=6, num_attention_heads=2, intermediate_size=1024)
        self.hidden_dim = self.bert_config.hidden_size
        self.pad_token_id = tokenizer.pad_token_id
        self.pos_pad_token_id = tokenizer.pos_pad_token_id

        self.dorn = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
    
    def forward(self, x, pos_id, mask):
        # predict order
        x, score_mask = self.dorn(x), mask.unsqueeze(1)
        score = torch.bmm(x, torch.transpose(x, 1, 2))   # (# of shuffled x multi_turn x hidden_dim) * (# of shuffled x hidden_dim x multi_turn) = # of shuffled x multi_turn x multi_turn
        score = score.masked_fill(score_mask==self.pad_token_id, 0)
        score = torch.sum(score, dim=2) / torch.sum(score_mask, dim=2)   # mean except for padding
        score = score.masked_fill(mask==self.pad_token_id, float('-inf'))

        # gt score
        pos_max = torch.max(pos_id, dim=1, keepdim=True)[0]
        pos_id = (pos_max - pos_id) / pos_max
        pos_id = pos_id.masked_fill(mask==self.pad_token_id, float('-inf'))
        
        # get probs
        pred_order_prob = F.log_softmax(score, dim=-1)
        gt_order_prob = F.softmax(pos_id, dim=-1)

        return pred_order_prob, gt_order_prob



# NN for NUG
class NextUtteranceGeneration(nn.Module):
    def __init__(self, config, tokenizer):
        super(NextUtteranceGeneration, self).__init__()
        # self.bert_config = BertConfig.from_pretrained(config.pretrained_model)
        # self.bert_config.is_decoder = True
        # self.bert_config.add_cross_attention = True
        # self.model = BertLMHeadModel(self.bert_config)
        
        # self.model = BertLMHeadModel.from_pretrained(config.pretrained_model, is_decoder=True, add_cross_attention=True)
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size
        self.encoder_config = BertConfig(vocab_size=30522, hidden_size=256, num_hidden_layers=6, num_attention_heads=2, intermediate_size=1024)
        self.encoder_config.is_decoder = True
        self.encoder_config.add_cross_attention = True
        self.model = BertLMHeadModel(self.encoder_config)


        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.bert_config.hidden_size, nhead=self.bert_config.num_attention_heads)
        # self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.bert_config.num_hidden_layers)
        # self.emb_layer = nn.Embedding(self.vocab_size, self.bert_config.hidden_size)
        # self.fc_layer = nn.Linear(self.bert_config.hidden_size, self.vocab_size)

    def make_pad_mask(self, x, id):
        mask = torch.where(x==id, 0, 1)
        return mask

    
    def forward(self, x, trg, cntx_pad_mask):
        trg_pad_mask = self.make_pad_mask(trg, self.pad_token_id)
        output = self.model(
            input_ids=trg,
            attention_mask=trg_pad_mask,
            encoder_hidden_states=x,
            encoder_attention_mask=cntx_pad_mask)

        # trg = self.emb_layer(trg)
        # output = self.transformer_decoder(trg.transpose(0, 1), x.transpose(0, 1))#, tgt_mask=trg_pad_mask.transpose(0,1).bool(), memory_mask=cntx_pad_mask.transpose(0,1).bool())
        # output = self.fc_layer(output.transpose(0, 1))

        return output.logits



# DialogBERT
class DialogBERT(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(DialogBERT, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.pos_pad_token_id = tokenizer.pos_pad_token_id
        self.device = device

        self.utteranceEnc = UtteranceEncoder(self.config)
        self.cntxEnc = ContextEncoder(self.config)
        self.mur = MaskedUtteranceRegression(self.config)
        self.duor = DistributedUtteranceOrderRanking(self.config, self.tokenizer)
        self.nug = NextUtteranceGeneration(self.config, self.tokenizer)


    def make_pad_mask(self, x, id):
        mask = torch.where(x==id, 0, 1)
        return mask


    def select_MUR_id(self, lm_trg):
        # to select only MUR-ed data
        mur_id = torch.sum(lm_trg, dim=2)   # B x multi_turn x max_len  ->  B x multi_turn
        mur_id = mur_id > 0     # B x multi_turn x max_len  ->  B x multi_turn
        return mur_id


    def select_DUOR_id(self, shuffled, cntx_pad_mask):
        con1 = shuffled == 1   # to select shuffled data
        con2 = torch.sum(cntx_pad_mask.cpu(), dim=-1) >= 4   # to select data that contain more than 2 dialogues except for CLS, SEP parts
        duor_id = torch.logical_and(con1, con2)
        return duor_id
    
    
    def select_NUG_id(self, shuffled):
        nug_id = shuffled == 0   # to avoid shuffled data
        return nug_id


    def forward(self, src, trg, lm_trg, pos_id, shuffled, phase):
        mur_output, mur_trg, duor_output, duor_trg = None, None, None, None

        # make mask
        uttn_pad_mask = self.make_pad_mask(src, self.pad_token_id).to(self.device)
        cntx_pad_mask = self.make_pad_mask(pos_id, self.pos_pad_token_id).to(self.device)

        # utterance and context encoding
        output = self.utteranceEnc(src, uttn_pad_mask)
        # uttn_pad_mask = uttn_pad_mask.view(uttn_pad_mask.size(0), -1)
        output = self.cntxEnc(output, cntx_pad_mask)   # B x multi_turn x hidden_dim 
        
        if phase == 'train':
            # for MUR, MUR target must be detached and had False requires_grad
            with torch.no_grad():  
                mur_id = self.select_MUR_id(lm_trg)
                mur_trg, mur_pad_mask = lm_trg[mur_id].detach(), uttn_pad_mask[mur_id].detach()   # (# of mur-ed sentence) x max_len
                mur_trg = self.utteranceEnc.model.bert(input_ids=mur_trg, attention_mask=mur_pad_mask)['pooler_output']   # (# of mur-ed sentence) x hidden_dim
            mur_output = self.mur(output[mur_id])   # (# of mur-ed sentence) x hidden_dim

            # for DUOR
            duor_id = self.select_DUOR_id(shuffled, cntx_pad_mask)
            duor_output, duor_trg = self.duor(output[duor_id], pos_id[duor_id], cntx_pad_mask[duor_id])
        
        # for NUG
        nug_id = self.select_NUG_id(shuffled)
        # nug_output = self.nug(output[nug_id], trg[nug_id], uttn_pad_mask[nug_id])
        nug_output = self.nug(output[nug_id], trg[nug_id], cntx_pad_mask[nug_id])
        
        return (nug_output, trg[nug_id]), (mur_output, mur_trg), (duor_output, duor_trg)