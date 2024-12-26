import time,os.path,sys
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import  ImageEncoder, build_transformer
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
if os.path.exists('input.txt'):
    sys.stdin = open( 'input.txt','r')
    sys.stdout = open('output.txt','w')

# Data parameters
data_folder = '/content/caption_data/'  # folder with data files saved by create_input_files.py
data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 12  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 0  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none

def beam_search_decode(decoder, beam_size, imgs, word_map,  max_len, device,alpha=0.7):
    sos_idx = word_map['<start>']
    eos_idx = word_map['<end>']
    # Precompute the encoder output and reuse it for every step
    encoder_output =decoder.encode(imgs)
    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.empty(1, 1).fill_(sos_idx).type_as(imgs).to(device)
    # Create a candidate list    sequence, score, length
    candidates = [(decoder_initial_input, 0,1)]
    while True:
        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _,_ in candidates]):
            break
        # Create a new list of candidates
        new_candidates = []

        for candidate, score,length in candidates:

            # Do not expand candidates that have reached the eos token
            if candidate[0][-1].item() == eos_idx:
                new_candidates.append((candidate, score, length))
                continue
            # Build the candidate's mask
            candidate_mask = causal_mask(candidate.size(1)).to(device)
            # calculate output
            out =decoder.decode(encoder_output,candidate, candidate_mask)
            # get next token probabilities
            prob = decoder.projection(out[:, -1])
            # get the top k candidates
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            for i in range(beam_size):
                # for each of the top k candidates, get the token and its probability
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                # create a new candidate by appending the token to the current candidate
                new_candidate = torch.cat([candidate, token], dim=1)
                # We sum the log probabilities because the probabilities are in log space
                new_score = score + token_prob
                new_candidates.append((new_candidate, new_score, length + 1))

        # Apply length normalization to scores
        new_candidates = [
            (seq, score / (length ** alpha), length)  # Normalize score
            for seq, score, length in new_candidates
        ]
        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]

        # If all the candidates have reached the eos token, stop
        if all([cand[0][-1].item() == eos_idx for cand, _,_ in candidates]):
            break
    # Return the best candidate
    return candidates[0][0].squeeze()

def greedy_decode(decoder, imgs,  word_map, max_len, device):
    sos_idx = word_map['<start>']
    eos_idx = word_map['<end>']

    # Precompute the encoder output and reuse it for every step
    encoder_output = decoder.encode(imgs)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(imgs).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)

        # calculate output
        out = decoder.decode(encoder_output, decoder_input, decoder_mask)


        # get next token
        prob = decoder.projection(out[:, -1])
        _, next_word = torch.max(prob, dim=1) # max_value, and its index for each row 
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(imgs).fill_(next_word.item()).to(device)], dim=1)
            #[[1]]--->[[1,2,15,20,6]] as tensor (batch,seq_len)

        if next_word == eos_idx:
            break

    return decoder_input # [1,2,15,20,6] tensor if .squeeze(0) 

def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map
    filename = 'checkpoint_' + data_name + '.pth.tar'
    if os.path.exists(filename):
        print(f"Loading checkpoint '{filename}'...")
        checkpoint = torch.load(filename)
    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder=build_transformer(7003,196)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = ImageEncoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1).to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)
    
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 =validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion, word_map=word_map)

               

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        recent_bleu4=0
        is_best=0
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):


    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    # batch_time = AverageMeter()  # forward prop. + back prop. time
    # data_time = AverageMeter()  # data loading time
    # losses = AverageMeter()  # loss (per word decoded)
    # top5accs = AverageMeter()  # top5 accuracy

    # start = time.time()

    # Batches
    batch_iterator = tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}")

    for i, (imgs, caps, labels,decoder_mask,caplen) in enumerate(batch_iterator):
        # data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)  #b,3,h,h
        caps = caps.to(device)  #b,seq
        labels = labels.to(device)
        decoder_mask=decoder_mask.to(device)
        caplen=caplen.to(device)


        # Forward prop.


        imgs = encoder(imgs)
        encoder_output=decoder.encode(imgs)
        decoder_output =decoder.decode(encoder_output, caps, decoder_mask)
        proj_output = decoder.projection(decoder_output) 





        # Calculate loss
        # loss = criterion(scores, targets)
        loss = criterion(proj_output.view(-1,7003), labels.view(-1))
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})



        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 )


def validate(val_loader, encoder, decoder, criterion,word_map):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()


    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    max_len=196
    reverse_word_map = {v: k for k, v in word_map.items()}
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, labels,allcaps,decoder_mask,caplen) in enumerate(val_loader):
            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            labels = labels.to(device)
            allcaps=allcaps.to(device)
            decoder_mask=decoder_mask.to(device)
            caplen=caplen.to(device)
            
            imgs=encoder(imgs)

            Image_Caption = greedy_decode(decoder, imgs,  word_map, max_len, device) # (batch,seq_len)
            preds=Image_Caption.tolist()
            model_out=Image_Caption.squeeze(0) 

            decoded_words = [reverse_word_map[token.item()] for token in model_out]
            model_out_text=' '.join([word for word in decoded_words if word not in ['<start>', '<end>', '<pad>']])

            labels=labels.view(-1)
            decoded_labels=[reverse_word_map[token.item()] for token in labels]
            label_text=' '.join([word for word in decoded_labels if word not in ['<start>', '<end>', '<pad>']])
            print(f"{f'TARGET: ':>12}{label_text}")
            print(f"{f'PREDICTED: ':>12}{model_out_text}")

            # loss = criterion(scores, targets)

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      )

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
        
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses

            for j, p in enumerate(preds):  # Loop over each sequence in the batch
                hypothesis = p[:caplen[j].item()]  # Trim predictions to the valid length
                hypothesis = [
                    w for w in hypothesis if w not in {word_map['<start>'], word_map['<pad>']}
                ]  # Remove <start> and <pad>
                if word_map['<end>'] in hypothesis:  # Truncate at <end> if present
                    hypothesis = hypothesis[:hypothesis.index(word_map['<end>'])]
                hypotheses.append(hypothesis)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                bleu=bleu4))

    return bleu4






if __name__ == '__main__':
    main()

