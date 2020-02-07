import shutil
import time
from tensorboardX import SummaryWriter #sasaki
import torch
from torch.autograd import Variable
from IPython.core.debugger import Pdb
import json

def train(model, dataloader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    example_count = 0
    step = 0
    # Iterate over data.
    for questions, images, image_ids, answers, ques_ids in dataloader:
        questions = questions.to(device)
        images = images.to(device)
        image_ids = image_ids.to(device)
        answers = answers.to(device)
        
        questions, answers = Variable(questions).transpose(0, 1), Variable(answers)  #sasaki

        # zero grad
        optimizer.zero_grad()
        ans_scores = model(images, questions, image_ids)
        _, preds = torch.max(ans_scores, 1)
        loss = criterion(ans_scores, answers)

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum((preds == answers).data)
        example_count += answers.size(0)
        step += 1
        if step % 5000 == 0:
            print('running loss: {}, running_corrects: {}, example_count: {}, acc: {}'.format(
                running_loss / example_count, running_corrects, example_count, (float(running_corrects) / example_count) * 100))
        # if step * batch_size == 40000:
        #     break
    epoch_loss = running_loss / example_count
    acc = (float(running_corrects) / example_count) * 100
    print('Train Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(epoch_loss,
                                                           acc, running_corrects, example_count))
    return epoch_loss, acc


def validate(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    example_count = 0
    # Iterate over data.
    for questions, images, image_ids, answers, ques_ids in dataloader:
        questions = questions.to(device)
        images = images.to(device)
        image_ids = image_ids.to(device)
        answers = answers.to(device)

        questions, answers = Variable(questions).transpose(0, 1), Variable(answers)

        # zero grad
        ans_scores = model(images, questions, image_ids)
        _, preds = torch.max(ans_scores, 1)
        loss = criterion(ans_scores, answers)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum((preds == answers).data)
        example_count += answers.size(0)
    epoch_loss = running_loss / example_count
    acc = (float(running_corrects) / example_count) * 100
    print('Validation Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(epoch_loss,acc, running_corrects, example_count))
    return epoch_loss, acc


def train_model(model, data_loaders, criterion, optimizer, scheduler, save_dir, num_epochs, device, best_accuracy, start_epoch):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = best_accuracy
    writer = SummaryWriter(save_dir)
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_begin = time.time()
        train_loss, train_acc = train(model, data_loaders['train'], criterion, optimizer, device)
        train_time = time.time() - train_begin
        print('Epoch Train Time: {:.0f}m {:.0f}s'.format(train_time // 60, train_time % 60))
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Accuracy', train_acc, epoch)

        validation_begin = time.time()
        val_loss, val_acc = validate(model, data_loaders['val'], criterion, device)
        validation_time = time.time() - validation_begin
        print('Epoch Validation Time: {:.0f}m {:.0f}s'.format(validation_time // 60, validation_time % 60))
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_acc, epoch)

        # deep copy the model
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_model_wts = model.state_dict()

        state = {'epoch':epoch, 
                 'best_acc':best_acc, 
                 'state_dict':model.state_dict(), 
                 "optimizer":optimizer.state_dict()}
        save_checkpoint(save_dir, state, is_best)

        writer.export_scalars_to_json(save_dir + "/all_scalars.json")
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json(save_dir + "/all_scalars.json")
    writer.close()

    # load best model weights
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)

    return model


def save_checkpoint(save_dir, state, is_best):
    savepath = save_dir + '/checkpoint.pth.tar'
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, save_dir + '/model_best.pth.tar')

