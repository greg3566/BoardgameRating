# BoardgameRating
Personalized recommendation of boardgame with rating data from boardgamegeek.com

Blending of
1. Matrix Factorization
2. User-oriented Neighborhood filtering
3. Item-based (Categories, Mechanics, Number of Players, Weight(complexity) ) filtering

For 1,2,
 Gather user names who rated top boardgames recently.
 Gather games which is rated by gathered username.
 Use only rating information between does users and games.
For 3
 Information from 1,2 + information of games provided by boardgamegeek.
All information was gathered by API.
