
# coding: utf-8

# In[6]:


pos_f = open('test_pos_result')
pos_dat = pos_f.readlines()
pos_f.close()
neg_f = open('test_neg_result')
neg_dat = neg_f.readlines()
neg_f.close()
pos_dt = []
for i in pos_dat:
    if i[0] == 't':
        new = [i[7:]]
    elif i[0] == 'p':
        new.append(i[6:])
    else:
        pos_dt.append(new)
pos_dt = sorted(pos_dt, key = lambda x:x[0])
neg_dt = []
for i in neg_dat:
    if i[0] == 't':
        new = [i[7:]]
    elif i[0] == 'p':
        new.append(i[6:])
    else:
        neg_dt.append(new)
neg_dt = sorted(neg_dt, key = lambda x:x[0])
pred_pos = [i[1].split() for i in pos_dt]
pred_neg = [i[1].split() for i in neg_dt]
preds = {'positive': pred_pos, 'negative': pred_neg}
