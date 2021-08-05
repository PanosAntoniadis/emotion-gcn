import torch


def train_model(dataloader, model, criterion_cat, criterion_cont, optimizer):
    model.train()
    running_loss = 0.0
    running_loss_cat = 0.0
    running_loss_cont = 0.0
    y_pred = []
    y_true = []

    device = next(model.parameters()).device
    for index, batch in enumerate(dataloader, 1):
        inputs, labels, labels_cont, inp = batch

        inputs = inputs.to(device)
        labels = labels.to(device)
        labels_cont = labels_cont.to(device)
        inp = inp.to(device)

        optimizer.zero_grad()
        outputs_cat, outputs_cont = model(inputs, inp)

        loss_cat = criterion_cat(outputs_cat, labels)
        loss_cont = criterion_cont(outputs_cont.double(), labels_cont.double())
        _, preds = torch.max(outputs_cat, 1)
        loss = loss_cat + loss_cont

        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        running_loss_cat += loss_cat.data.item()
        running_loss_cont += loss_cont.data.item()

        y_pred.append(preds.cpu().numpy())
        y_true.append(labels.cpu().numpy())

    return running_loss / index, running_loss_cat / index, running_loss_cont / index, (y_true, y_pred)


def eval_model(dataloader, model, criterion_cat, criterion_cont):
    model.eval()
    running_loss = 0.0
    running_loss_cat = 0.0
    running_loss_cont = 0.0
    y_pred = []
    y_true = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            inputs, labels, labels_cont, inp = batch

            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_cont = labels_cont.to(device)
            inp = inp.to(device)

            outputs_cat, outputs_cont = model(inputs, inp)

            loss_cat = criterion_cat(outputs_cat, labels)
            loss_cont = criterion_cont(
                outputs_cont.double(), labels_cont.double())
            _, preds = torch.max(outputs_cat, 1)
            loss = loss_cat + loss_cont

            y_pred.append(preds.cpu().numpy())
            y_true.append(labels.cpu().numpy())

            running_loss += loss.data.item()
            running_loss_cat += loss_cat.data.item()
            running_loss_cont += loss_cont.data.item()

    return running_loss / index, running_loss_cat / index, running_loss_cont / index, (y_true, y_pred)
