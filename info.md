# Info

This file serves as a place for me to keep notes and papers regarding this research, so it's all in one place

## Models to compare
- RNN
- ANN
- Clustering (time permitting)
- Random Forest
- KNN
- Label spreading

## Questions to answer
- How should I encode a chess game for the models?
- Can label spreading be used for prediction as well as giving labels to unlabeled data?
- What should be the balance of the full dataset? Chess.com claims to detect cheating [in about 0.6%](https://www.chess.com/cheating) of accounts
- How does lichess.org utilize [CNNs](https://github.com/lichess-org/kaladin) to detect cheating?
- Lichess also uses [irwin](https://github.com/clarkerubber/irwin) for cheat detection, using PV (principle variation, the sequence of best moves), winning chances, and move times

## Links to papers
- [Cheat Detection on Online Chess Games using Convolutional and Dense Neural Network](https://ieeexplore.ieee.org/abstract/document/9702792?casa_token=-c-BLJIX9J0AAAAA:HGyjtJAj34QY7GtJ0uNWtHN9hnJVVEJ94owfNBPZKMrOHhiepeZBLeXb8myc9ZHbhJt4kfJs20SM)
- [Towards Transparant Cheat Detection in Online Chess](https://books.google.com/books?hl=en&lr=&id=0HfAEAAAQBAJ&oi=fnd&pg=PA163&dq=chess+cheat+detection&ots=ox0or3iH63&sig=9LEpJkcIBjW2c-FO-2qF1HwzOG0#v=onepage&q=chess%20cheat%20detection&f=false)
- [Detecting Fair Play Violations in Chess Using Neural Networks](https://ceur-ws.org/Vol-3885/paper13.pdf)
- [Towards Transparent Cheat Detection in Online Chess: An Application of Human and Computer Decision-Making Preferences](https://link.springer.com/chapter/10.1007/978-3-031-34017-8_14)
- [The Impact of Artificial Intelligence on the Chess World](https://games.jmir.org/2020/4/e24049/)
- [Cheat Detection on Online Chess Games using Convolutional and Dense Neural Network](https://ieeexplore.ieee.org/abstract/document/9702792?casa_token=QjdfMdLHq3AAAAAA:sHHeAdcYVcjMpVz7X9PqEHEdLPAe7JCfRWEmYWCS_SBZYQ1wnuYleHK-oKU2zcB-AlCxwImJy7BN)
- [CLASSIFICATION OF CHESS GAMES: An exploration of classifiers for anomaly detection in chess](https://www.proquest.com/openview/58374853af3a795f3c963971d7a72f5c/1?pq-origsite=gscholar&cbl=18750&diss=y)
- [A Neural Network Approach to Chess Cheat Detection](https://books.google.com/books?hl=en&lr=&id=ui9NEQAAQBAJ&oi=fnd&pg=PA131&dq=chess+cheat+detection&ots=elpn_ApiMF&sig=HkI-GeaC9wVSXvo180YrCqv_K6c#v=onepage&q=chess%20cheat%20detection&f=false)

## Links to datasets
- [Labeled dataset](https://www.kaggle.com/datasets/brieucdandoy/chess-cheating-dataset) using Maia and rating from opening to create non-cheated moves, and Stockfish to create cheated moves
- [Technically labeled](https://www.kaggle.com/datasets/lichess/tournament-chess-games) (too big to store on Github)
- [Technically labeled](https://www.kaggle.com/datasets/nuezzz/chess-dataset)
- [Unlabeled](https://www.kaggle.com/datasets/arevel/chess-games) (too big to store on Github)

Note that "Technically labeled" means that games within this dataset were played at in-person tournaments. Since there is in-person proctoring at these tournaments, those games could be considered to be completely fair, and contain no moves that are cheated. 
- In this repo, labeled data files starts with l, technically labeled starts with t, and unlabeled starts with u

### Labeled dataset info
- In "Liste cheat white" column, 0s correspond to white's moves that are not cheated, while 1's refer to moves that are cheated. Similar for "Liste cheat black", except for black's moves rather that white's
- Games are in PGN format
- 48,933 total games, 12,278 with no cheater at all, 11,028 with both sides cheating, and 25,627 where only one side cheats