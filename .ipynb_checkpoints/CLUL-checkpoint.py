class CLUL:
    def __init__(self,batch_size = 16,lr = 1e-3,memory_size = 500,
                 forget_size = 100,epoch=5**kwargs,loss ='focal_loss',
                 total_class_num = 66):
        feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
        whisper_model = whisper.load_model("small")
        self.model = Audio_Encoder(feature_extractor, whisper_model)
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.logger = logging.getLogger()
        self.forget_list = []
        self.memory_list = []
        self.memory_size = memory_size
        self.forget_size = forget_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.criterion = get_loss_func(loss)
        self.evaluator = Evaluator(model, self.num_pretrain_class, self.device)
        self.cltask = {
            'task0':[15, 17, 60, 50, 32, 24, 63, 36, 31, 40, 52, 4, 25],
            "task1" = [48, 54, 35, 62, 13, 42, 37, 49, 51, 45, 44, 14, 5],
            "task2" = [46, 18, 57, 28, 11, 30, 61, 27, 22, 2, 29, 0, 19],
            "task3" = [3, 59, 10, 12, 8, 1, 26, 23, 34, 58, 64, 56, 41],
            "task4" = [47, 20, 53, 39, 9, 21, 16, 38, 33, 43, 6, 7, 55]
        }
        self.ultask = {
            "un_task1" = [15,17,60],
            "un_task2" = [48, 54, 35],
            "un_task3" = [46, 18, 57],
            "un_task4" = [3, 59, 10],
            "un_task5" = [47, 20, 53]
        }
        
        
        
    def train(self,cur_iter):
        streamed_list,test_list = get_datalist(cur_iter)
        train_list = self.streamed_list + self.memory_list
        random.shuffle(train_list)
        train_loader, test_loader = get_dataloader(self.batch_size, self.n_worker, train_list, test_list)

        logger.info(f"Streamed samples: {len(streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)}")
        logger.info(f"Test samples: {len(test_list)}")
        # logger.info(f"Model: {self.model}")
        logger.info(f"Optimizer: {self.optimizer}")
        acc_list = []
        best = {'acc': 0, 'epoch': 0,'f1_score':0}

        for epoch in range(n_epoch):
            mean_loss = 0
            for batch_data_dict in tqdm(train_loader):
                batch_data_dict['waveform'] = batch_data_dict['waveform']
                batch_data_dict['target'] = batch_data_dict['target'].to(self.device)

                # Forward
                self.model.train()

                batch_output_dict = self.model(batch_data_dict['waveform'])
                """{'clipwise_output': (batch_size, classes_num), ...}"""
                batch_target_dict = {'target': batch_data_dict['target']}
                """{'target': (batch_size, classes_num)}"""
                # Loss

                loss = self.criterion(batch_output_dict, batch_target_dict)
                logger.info(f'Batch Training Initial Loss: {loss}')
                # Backwards
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss = loss.item()

                mean_loss += loss
            epoch_loss = mean_loss / len(train_loader)
            logger.info(f'Epoch {epoch} | Training Loss: {epoch_loss}')
            # Evaluate
            test_statistics = self.evaluator.evaluate(test_loader)
            ave_f1_score = np.mean(test_statistics['f1_score'])
            ave_acc = np.mean(test_statistics['accuracy'])
            acc_list.append(ave_acc)
            logger.info(f"Epoch {epoch} | Evaluation Accuracy: {ave_acc}|Evaluation f1_score: {ave_f1_score}")
            logger.info(f'Current Accuracy: {ave_acc} in epoch {epoch}.|Current f1_score: {ave_f1_score} in epoch {epoch}.')

            if ave_f1_score > best['f1_score']:
                best['acc'] = ave_acc
                best['f1_score'] = ave_f1_score
                best['epoch'] = epoch
                logger.info(f'Best Accuracy: {ave_acc} in epoch {epoch}.|Best f1_score: {ave_f1_score} in epoch {epoch}.')
                selected_state_dict = {}
                for name, param in self.model.named_parameters():
                    if 'projector' in name or 'classifier' in name or 'fc' in name and ('encoder' not in name):
                        selected_state_dict[name] = param
                torch.save(selected_state_dict,'/home/user/SED_Adaptation_Classifier-main/workspace/ref_youtube/{}/iter{}epoch.pt'.format(self.mem_manage,iteration))
                self.counter = 0
            else:
                self.counter += 1
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}.')
                if self.counter >= self.patience:
                    break
        return 
    def change_model(self, path):
        checkpoint_dict = torch.load(path)
        for name, param in self.model.named_parameters():
            if name in checkpoint_dict:
                param.data = checkpoint_dict[name]
                
    def equal_class_sampling(self, samples, num_class):
        class_list = [self.cltask["task0"], self.cltask["task1"],self.cltask["task2"],self.cltask["task3"],self.cltask["task4"]]
        cur_class_list = []
        for i in range(num_class//13):
            cur_class_list += class_list[i]
        mem_per_cls = self.memory_size // num_class
        sample_df = pd.DataFrame(samples)

        # Warning: assuming the classes were ordered following task number.
        ret = []
        for y in cur_class_list:
            cls_df = sample_df[(sample_df["category"].map(ytvos_category_dict)) == y]
            ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                orient="records"
            )

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            ret += (
                sample_df[~sample_df.exp.isin(pd.DataFrame(ret).exp)]
                .sample(n=num_rest_slots)
                .to_dict(orient="records")
            )

        num_dups = pd.DataFrame(ret).exp.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret

    def get_data(self, infer_loader, augment):
        Z, Z_, predict_list = [], [], []
        self.model.eval()
        with torch.no_grad():
            for id, data in enumerate(infer_loader):
                wavs = data['waveform']
                aug_wavs = []
                for wav in wavs:
                    aug_wav = augment(wav.unsqueeze(0).unsqueeze(0), sample_rate=1600)
                    aug_wavs.append(torch.as_tensor(aug_wav.squeeze(0).squeeze(0), dtype=torch.float32))

                output_dict = self.model(data['waveform'])
                aug_output_dict = self.model(aug_wavs)

                for z, z_ in zip(output_dict['feature'], aug_output_dict['feature']):
                    Z.append(z)
                    Z_.append(z_)

                clipwise_output = output_dict['clipwise_output']
                pres = np.argmax(clipwise_output.detach().cpu(), axis=1)
                target = np.argmax(data['target'].cpu(), axis=1)

                for pre in pres: predict_list.append(pre.item())

            class_label_dic = self.save_indexes(predict_list)
        return Z, Z_, class_label_dic, predict_list
    def save_indexes(self,arr):
        index_dict = {}
        for idx, num in enumerate(arr):
            if num in index_dict:
                index_dict[num].append(idx)
            else:
                  index_dict[num] = [idx]
        return index_dict

    def class_infoNCE(self, Z, Z_, class_label_dic, predict_list, temperature):
        ## You can change the method to calculate NCEs
        NCEs = []
        # print('This is cclass_label_dic',class_label_dic)
        for id in range(len(predict_list)):
            label = predict_list[id]
            same_label_list = class_label_dic[label]
            class_z = [Z[i] for i in same_label_list if i != id]
            class_z_ = [Z_[i] for i in same_label_list]

            positive_pair = class_z + class_z_

            positive_similarities = F.cosine_similarity(Z[id].unsqueeze(0), torch.stack(positive_pair)) / 2 + 0.5
            # print('This is postitive pair info',Z[id].unsqueeze(0).shape,torch.stack(positive_pair).shape,positive_similarities.shape)
            positive_value = torch.exp(positive_similarities / temperature).sum() / len(positive_pair)
            # print(positive_similarities,positive_value)
            neg_labels = [i for i in list(class_label_dic.keys()) if i != label]

            negative_values = 0
            for neg_label in neg_labels:
                neg_label_list = class_label_dic[neg_label]
                neg_z = [Z[i] for i in neg_label_list]
                neg_z_ = [Z_[i] for i in neg_label_list]
                negative_pair = neg_z + neg_z_
                negative_similarities = F.cosine_similarity(Z[id].unsqueeze(0), torch.stack(negative_pair)) / 2 + 0.5
                # print('This is negative pair info',Z[id].unsqueeze(0).shape,torch.stack(negative_pair).shape,negative_similarities.shape,len(negative_pair))
                negative_value = torch.exp(negative_similarities / temperature).sum() / len(negative_pair)
                # print(negative_similarities,negative_value)
                negative_values += negative_value

            NCE = -torch.log(positive_value / (positive_value + negative_values))
            # print('positive_value',positive_value,'negative values', negative_values,'this is single nce',NCE)
            NCEs.append(NCE)
        print(torch.stack(NCEs).shape)
        return torch.stack(NCEs)
    
    def single_mutual_info_sampling(self, candidates, cur, num_class):
        from audiomentations import Compose, Gain, AddGaussianNoise, PitchShift,TimeStretch,Shift
        from collections import Counter
        
        class_list = [self.cltask["task0"],self.cltask["task1"],self.cltask["task2"],self.cltask["task3"],self.cltask["task4"]]

        ulclass_list =   [None,self.ultask["task1"],self.ultask["task2"],self.ultask["task3"],self.ultask["task4"]]
        cur_class_list = []
        for i in range(num_class // 13):
            cur_class_list |= set(class_list[i])
            cur_class_list -= set(ulclass_list[i])
            

        # Unlearning Part:class deleted will not be added into the memory bank

        infer_df = pd.DataFrame(candidates)

        class_count = Counter(infer_df['category'])
        print('Before Unpdate Statistics')
        for name, number in class_count.items():
            print(name, number)
        # mem_per_cls = self.memory_size // num_class  # kc: the number of the samples of each class

        batch_size = 8
        temperature = 0.05
        ret = []
        infer_loader = get_dataloader(infer_df, 'ref_youtube_audio', split='test', batch_size=batch_size, num_class=num_class,
                                      num_workers=8)
        augment = Compose([
            # Gain(min_gain_in_db=-12.0, max_gain_in_db=12.0),
            # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.001),
            PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.5),
            # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015),
            # TimeShift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            # Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
            # TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
        ])

        Z, Z_, class_label_dic, predict_list = self.get_data(infer_loader, augment)
        assert (len(Z) == len(Z_) == len(predict_list))

        cur_NCEs = self.class_infoNCE(Z, Z_, class_label_dic, predict_list, temperature)

        path = '/home/user/SED_Adaptation_Classifier-main/workspace/ref_youtube/MIO/iter{}epoch.pt'.format(cur - 1)
        self.change_model(path)

        pre_Z, pre_Z_, pre_class_label_dic, pre_predict_list = self.get_data(infer_loader, augment)
        assert (len(Z) == len(Z_) == len(predict_list))

        pre_NCEs = self.class_infoNCE(pre_Z, pre_Z_, pre_class_label_dic, pre_predict_list, temperature)

        path = '/home/user/SED_Adaptation_Classifier-main/workspace/ref_youtube/MIO/iter{}epoch.pt'.format(cur)
        self.change_model(path)

        # print(len(Z),len(Z_),len(predict_list),len(candidates))

        NCEs = pre_NCEs - cur_NCEs
        for candidate,NCE in zip(candidates,NCEs):candidate['NCE'] = NCE

        sample_df = pd.DataFrame(candidates)

        mem_per_cls = self.memory_size // cur_class_list  # kc: the number of the samples of each class


        for i in cur_class_list:
            cls_df = sample_df[(sample_df["category"].map(ytvos_category_dict)) == i]
            if len(cls_df) <= mem_per_cls:
                ret += cls_df.to_dict(orient="records")
            else:
                jump_idx = len(cls_df) // mem_per_cls
                uncertain_samples = cls_df.sort_values(by="NCE")[::jump_idx]
                ret += uncertain_samples[:mem_per_cls].to_dict(orient="records")

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            ret += (
                sample_df[~sample_df.exp.isin(pd.DataFrame(ret).exp)]
                .sample(n=num_rest_slots)
                .to_dict(orient="records")
            )

        num_dups = pd.DataFrame(ret).exp.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")


        class_count = Counter(pd.DataFrame(ret)['category'])
        print('After Unpdate Statistics')
        for name, number in class_count.items():
            print(name, number)

        return ret
     def double_mutual_info_sampling(self, candidates, cur, num_class):
        from audiomentations import Compose, Gain, AddGaussianNoise, PitchShift,TimeStretch,Shift
        from collections import Counter
        
        ulclass_list =   [None,self.ultask["task1"],self.ultask["task2"],self.ultask["task3"],self.ultask["task4"]]
        class_list = [self.cltask["task0"], self.cltask["task1"],self.cltask["task2"],self.cltask["task3"],self.cltask["task4"]]
        cl_class_list = []
        ul_class_list = []
        for i in range(num_class // 13):
            cur_class_list |= set(class_list[i])
            cur_class_list -= set(ulclass_list[i])
        cur_class_list.add(self.total_class_num-1)
        # Unlearning Part:class deleted will not be added into the memory bank

        infer_df = pd.DataFrame(candidates)

        class_count = Counter(infer_df['category'])
        print('Before Unpdate Statistics')
        for name, number in class_count.items():
            print(name, number)
        # mem_per_cls = self.memory_size // num_class  # kc: the number of the samples of each class

        batch_size = 8
        temperature = 0.05
        ret = []
        infer_loader = get_dataloader(infer_df, 'ref_youtube_audio', split='test', batch_size=batch_size, num_class=num_class,
                                      num_workers=8)
        augment = Compose([
            # Gain(min_gain_in_db=-12.0, max_gain_in_db=12.0),
            # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.001),
            PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.5),
            # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015),
            # TimeShift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            # Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
            # TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
        ])

        Z, Z_, class_label_dic, predict_list = self.get_data(infer_loader, augment)
        assert (len(Z) == len(Z_) == len(predict_list))

        cur_NCEs = self.class_infoNCE(Z, Z_, class_label_dic, predict_list, temperature)

        path = '/home/user/SED_Adaptation_Classifier-main/workspace/ref_youtube/MIO/iter{}epoch.pt'.format(cur - 1)
        self.change_model(path)

        pre_Z, pre_Z_, pre_class_label_dic, pre_predict_list = self.get_data(infer_loader, augment)
        assert (len(Z) == len(Z_) == len(predict_list))

        pre_NCEs = self.class_infoNCE(pre_Z, pre_Z_, pre_class_label_dic, pre_predict_list, temperature)

        path = '/home/user/SED_Adaptation_Classifier-main/workspace/ref_youtube/MIO/iter{}epoch.pt'.format(cur)
        self.change_model(path)

        # print(len(Z),len(Z_),len(predict_list),len(candidates))

        NCEs = pre_NCEs - cur_NCEs
        for candidate,NCE in zip(candidates,NCEs):candidate['NCE'] = NCE

        sample_df = pd.DataFrame(candidates)
         # kc: the number of the samples of each class in memory bank
        mem_per_cls = self.memory_size // (len(cur_class_list))
        
        for_per_cls = self.forget_size// len(
        


        for i in cur_class_list:
            cls_df = sample_df[(sample_df["category"].map(ytvos_category_dict)) == i]
            if len(cls_df) <= mem_per_cls:
                ret += cls_df.to_dict(orient="records")
            else:
                jump_idx = len(cls_df) // mem_per_cls
                uncertain_samples = cls_df.sort_values(by="NCE")[::jump_idx]
                ret += uncertain_samples[:mem_per_cls].to_dict(orient="records")

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            ret += (
                sample_df[~sample_df.exp.isin(pd.DataFrame(ret).exp)]
                .sample(n=num_rest_slots)
                .to_dict(orient="records")
            )

        num_dups = pd.DataFrame(ret).exp.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")


        # top_indices = np.argpartition(NCEs.cpu().numpy(), -2000)[-2000:]
        #
        # for index in top_indices:
        #     ret.append(candidates[index])

        class_count = Counter(pd.DataFrame(ret)['category'])
        print('After Unpdate Statistics')
        for name, number in class_count.items():
            print(name, number)

        return ret