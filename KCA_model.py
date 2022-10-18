import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_
class KCA(nn.Module):
    def __init__(self, args,input_dim, output_dim,ent_text_emb, ent_img_emb):
        super(KCA, self).__init__()
        h_dim = 100
        self.ent_text_emb, self.ent_img_emb = ent_text_emb, ent_img_emb
        self.args = args

        attention_dim = 50

        self.linear_text = nn.Linear(in_features=384, out_features=attention_dim, bias=True)
        self.linear_img = nn.Linear(in_features=383, out_features=attention_dim, bias=True)
        self.linear_text_project = nn.Linear(in_features=attention_dim*24, out_features=attention_dim, bias=True)
        self.linear_img_project = nn.Linear(in_features=attention_dim*4, out_features=attention_dim, bias=True)

        if args.double_entity_embedding:
            entity_dim = args.hidden_dim * 2
        else:
            entity_dim = args.hidden_dim
        if args.double_relation_embedding:
            relation_dim = args.hidden_dim * 2
        else:
            relation_dim = args.hidden_dim
        self.linear_rel1 = nn.Linear(in_features=relation_dim, out_features=24*4, bias=True)
        self.linear_rel2 = nn.Linear(in_features=relation_dim, out_features=24*4, bias=True)

        self.linear1 = nn.Linear(in_features=entity_dim + relation_dim, out_features=h_dim, bias=True)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=entity_dim, bias=True)
        self.relu = nn.LeakyReLU(0.1)
        self.layernorm = nn.LayerNorm(attention_dim)
        self.batch_index = torch.arange(self.args.batch_size)
        self.init_network()

    def init_network(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
            elif isinstance(m, nn.Embedding):
                xavier_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                xavier_normal_(m.weight)


    def forward(self, kge_model,positive_sample,mode,temperature,pre_sample,pos_neg_mask):
        if mode=='head-batch':
            h_or_t = 0
        elif mode=='tail-batch':
            h_or_t = 2

        relation_emb = kge_model.relation_embedding[positive_sample[:,1]]  #B x dim
        batchsize = relation_emb.size(0)

        if self.args.pre_sample_num:
            text_emb = self.relu(self.linear_text(self.ent_text_emb[pre_sample]))  # e x 4 x 200
            img_emb = self.relu(self.linear_img(self.ent_img_emb[pre_sample]))  # e x 24 x 200
        else:
            text_emb = self.relu(self.linear_text(self.ent_text_emb))  # e x 4 x 200
            img_emb = self.relu(self.linear_img(self.ent_img_emb))  # e x 24 x 200


        num_entity = text_emb.size(0)
        cross_mat = torch.matmul(img_emb,text_emb.permute(0,2,1))  # e x 24 x 4


        #### Negative KCA
        img_att = torch.matmul(torch.softmax(cross_mat.permute(0,2,1),dim=2),img_emb)  # e x 4 x 200
        rel_guided_img = torch.sigmoid(self.linear_rel1(relation_emb)).view(batchsize,24,4) # B x 24 x 4
        rel_guided_img = torch.mul(rel_guided_img.unsqueeze(1).expand(-1,num_entity,-1,-1),cross_mat)  # B x e x 24 x 4
        img_att_rel_guided = torch.matmul(rel_guided_img.permute(0,1,3,2),img_emb) # B x e x 4 x 200
        img_att_all = self.layernorm(img_att_rel_guided) + self.layernorm(img_att)  # # B x e x 4 x 200


        text_att = torch.matmul(torch.softmax(cross_mat,dim=2),text_emb)  # e x 24 x 200
        rel_guided_text = torch.sigmoid(self.linear_rel2(relation_emb)).view(batchsize,24,4)
        rel_guided_text = torch.mul(rel_guided_text.unsqueeze(1).expand(-1,num_entity,-1,-1),cross_mat)  # B x e x 24 x 4
        text_att_rel_guided = torch.matmul(rel_guided_text,text_emb)  # B x e x 24 x 200
        text_att_all = self.layernorm(text_att_rel_guided) + self.layernorm(text_att) # # B x e x 24 x 200

        #####
        img_att_all = self.linear_img_project(img_att_all.view(batchsize,num_entity,-1))  #  B x Entity x dim
        text_att_all = self.linear_text_project(text_att_all.view(batchsize,num_entity,-1)) #  B x Entity x dim


        if self.args.pre_sample_num:
            t = kge_model.entity_embedding[pre_sample]  # Num_Entity x dim
        else:
            t = kge_model.entity_embedding  # Num_Entity x dim


        relation_emb = relation_emb.unsqueeze(1).expand(-1,num_entity,-1)
        t = t.unsqueeze(0).expand(batchsize,-1,-1)
        att = torch.cat([t,relation_emb],dim=2)
        att = self.relu(self.linear1(att))
        att = torch.sigmoid(self.linear3(att)) #  B x Entity x dim
        t_att = t * att  #  B x Entity x dim

        if self.args.pre_sample_num:
            pos_img_emb,pos_text_emb,pos_tail_emb = self.positive_KCA(kge_model,positive_sample,h_or_t)
        else:
            pos_tail_index = positive_sample[:,h_or_t]  # B 
            pos_img_emb = img_att_all[self.batch_index,pos_tail_index,:]  # B x dim
            pos_text_emb = text_att_all[self.batch_index,pos_tail_index,:] # B x dim
            pos_tail_emb = t_att[self.batch_index,pos_tail_index,:] # B x dim


        ### compute the similiarity. consine. 
        simil_img = torch.matmul(img_att_all,pos_img_emb.unsqueeze(-1)).squeeze(-1) # batchsize x nEntity
        simil_img = torch.div(simil_img,(torch.norm(img_att_all,2,dim=2)*torch.norm(pos_img_emb,2,dim=1).unsqueeze(1) ).detach() + 1e-10  )
        simil_text = torch.matmul(text_att_all,pos_text_emb.unsqueeze(-1)).squeeze(-1)
        simil_text = torch.div(simil_text,(torch.norm(text_att_all,2,dim=2)*torch.norm(pos_text_emb,2,dim=1).unsqueeze(1) ).detach() + 1e-10 )
        simil_t = torch.matmul(t_att,pos_tail_emb.unsqueeze(-1)).squeeze(-1)
        simil_t = torch.div(simil_t,(torch.norm(t_att,2,dim=2)*torch.norm(pos_tail_emb,2,dim=1).unsqueeze(1)).detach() + 1e-10 )


        simil_t_forsample = torch.softmax(simil_t/temperature,dim=1)  # batchsize x nEntity
        simil_img_forsample = torch.softmax(simil_img/temperature,dim=1)
        simil_text_forsample = torch.softmax(simil_text/temperature,dim=1)      
        return simil_t_forsample + simil_img_forsample + simil_text_forsample ,simil_img, simil_text,simil_t


    def positive_KCA(self,kge_model,positive_sample,h_or_t):
        relation_emb = kge_model.relation_embedding[positive_sample[:,1]]  # b x dim
        pos_tail_index = positive_sample[:,h_or_t]
        pos_tail_emb = kge_model.entity_embedding[pos_tail_index]  # b x dim

        text_emb = self.relu(self.linear_text(self.ent_text_emb[pos_tail_index]))  # b x 4 x 200
        img_emb = self.relu(self.linear_img(self.ent_img_emb[pos_tail_index]))   # b x 24 x 200
        cross_mat = torch.matmul(img_emb,text_emb.permute(0,2,1))  # b x 24 x 4
        
        batchsize = relation_emb.size(0)

        #### BGA
        img_att = torch.matmul(torch.softmax(cross_mat.permute(0,2,1),dim=2),img_emb)  # b x 4 x 200
        rel_guided_img = torch.sigmoid(self.linear_rel1(relation_emb)).view(batchsize,24,4) # B x 24 x 4
        rel_guided_img = torch.mul(rel_guided_img,cross_mat)  # B x 24 x 4
        img_att_rel_guided = torch.matmul(rel_guided_img.permute(0,2,1),img_emb) # B x 4 x 200
        img_att_all = self.layernorm(img_att_rel_guided) + (img_att)  # B x 4 x 200

        text_att = torch.matmul(torch.softmax(cross_mat,dim=2),text_emb)  # b x 24 x 200
        rel_guided_text = torch.sigmoid(self.linear_rel2(relation_emb)).view(batchsize,24,4)  # B x 24 x 4
        rel_guided_text = torch.mul(rel_guided_text,cross_mat)  # B x 24 x 4
        text_att_rel_guided = torch.matmul(rel_guided_text,text_emb)  # B x 24 x 200
        text_att_all = self.layernorm(text_att_rel_guided) + self.layernorm(text_att) # # B x 24 x 200

        img_att_all = self.linear_img_project(img_att_all.view(batchsize,-1))  #  B  x dim
        text_att_all = self.linear_text_project(text_att_all.view(batchsize,-1)) #  B  x dim


        att = torch.cat([pos_tail_emb,relation_emb],dim=1)
        att = self.relu(self.linear1(att))
        att = torch.sigmoid(self.linear3(att)) #  B x dim
        pos_tail_emb = pos_tail_emb * att

        return img_att_all,text_att_all,pos_tail_emb

