def logPredictions(dataset, predictions, gold, dsname, log=None):
    unified = []
    for i in range(len(dataset)):
        unified.append([dataset[i], predictions[i], gold[i], 0, 0, 0])

    join = lambda e,l: '%s:"%s"' % (e,l)

    max_left, max_right = 0, 0
    for (ent1, lbl1, ent2, lbl2, _) in dataset:
        left_joined = join(ent1, lbl1)
        right_joined = join(ent2, lbl2)
        if len(left_joined) > max_left: max_left = len(left_joined)
        if len(right_joined) > max_right: max_right = len(right_joined)

    # get gold indexing
    unified.sort(key=lambda k:k[2], reverse=True)
    for i in range(len(unified)):
        unified[i][4] = i
    # get pred indexing
    unified.sort(key=lambda k:k[1], reverse=True)
    for i in range(len(unified)):
        unified[i][3] = i
    # get errors
    for i in range(len(unified)):
        unified[i][5] = abs(unified[i][4] - unified[i][3])

    # write in gold order
    unified.sort(key=lambda k:k[2], reverse=True)
    log.writeln('\n\n== %s :: Gold order =============================================================================================' % dsname)
    for ((ent1, lbl1, ent2, lbl2, _), pred, gold, pred_ix, gold_ix, error) in unified:
        log.writeln(('%{0}s  %{1}s  -->  %f (%3d)  ||  %f (%3d)  [Error: %d]').format(max_left, max_right)
            % (join(ent1, lbl1), join(ent2, lbl2), pred, pred_ix, gold, gold_ix, error))

    # write in prediction order
    unified.sort(key=lambda k:k[1], reverse=True)
    log.writeln('\n\n== %s :: Predicted order ========================================================================================' % dsname)
    for ((ent1, lbl1, ent2, lbl2, _), pred, gold, pred_ix, gold_ix, error) in unified:
        log.writeln(('%{0}s  %{1}s  -->  %f (%3d)  ||  %f (%3d)  [Error: %d]').format(max_left, max_right)
            % (join(ent1, lbl1), join(ent2, lbl2), pred, pred_ix, gold, gold_ix, error))

    # get top 10 best and worst errors
    unified.sort(key=lambda k:k[5], reverse=True)
    log.writeln('\n\n== %s :: Worst errors ===========================================================================================' % dsname)
    for ((ent1, lbl1, ent2, lbl2, _), pred, gold, pred_ix, gold_ix, error) in unified[:10]:
        log.writeln(('%{0}s  %{1}s  -->  %f (%3d)  ||  %f (%3d)  [Error: %d]').format(max_left, max_right)
            % (join(ent1, lbl1), join(ent2, lbl2), pred, pred_ix, gold, gold_ix, error))
    unified.sort(key=lambda k:k[5], reverse=False)
    log.writeln('\n\n== %s :: Best errors ============================================================================================' % dsname)
    for ((ent1, lbl1, ent2, lbl2, _), pred, gold, pred_ix, gold_ix, error) in unified[:10]:
        log.writeln(('%{0}s  %{1}s  -->  %f (%3d)  ||  %f (%3d)  [Error: %d]').format(max_left, max_right)
            % (join(ent1, lbl1), join(ent2, lbl2), pred, pred_ix, gold, gold_ix, error))

    log.writeln()
