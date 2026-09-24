# fmi-steam-reviews

#

This is a project for the 2025/26 course of "Natural Language Processing Approaches" under the Faculty of Mathematics and Informatics (FMI), part of Sofia University "St. Kliment Ohridski".

---

The project goal is to perform sentiment analysis of game reviews from the popular game platform Steam. Two games have been chosen as targets - Concord, a controversial game with a total lifespan of two weeks, and Warframe, a long established title of 12 years. On top of that, we perform keyword extraction on the reviews, in order to see the most important topics in each review and in all reviews for the given game. That enables us to focus on the most important topics for each game's feedback and see whether a topics was positive or negative thanks to the previous sentiment analysis.

## Technologies

- sklearn
- numpy
- nltk
- rake-nltk - keyword extraction
- [LeXmo](https://github.com/dinbav/LeXmo)     - sentiment and emotion analysis of text

## Resources

Special thanks for the [Steam  Reviews  Analyze  Tool](https://docs.google.com/spreadsheets/d/1LpUzyYDHsGYL81ndIg8jZOB0PN-VXe7I3ppOGeBGm44/edit?gid=0#gid=0) developed by [Alexander  Blintsov](https://www.ablintsov.dev/), which greatly simplified the download of the reviews.
