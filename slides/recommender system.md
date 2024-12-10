Goals of a recommender system:
[[iml/slides/T11_2024_students_wide.pdf#page=19&selection=4,0,6,14|Help people make decisions]]
[[iml/slides/T11_2024_students_wide.pdf#page=19&selection=18,0,20,18|Help maintain awareness]]

> ([[iml/slides/T11_2024_students_wide.pdf#page=26|T11_2024_students_wide, p.26]])
> - **Knowledge-based recommender system (KB)** – knowledge about users and products used to reason what meets the user’s requirements, using discrimination tree, decision support tools, [[case-based reasoning (CBR)]]
> - **Collaborative filtering (CF)** – aggregation of consumers’ preferences and recommendations to other users based on similarity in behavioral patterns
> - **Content-based (CN)** – supervised machine learning used to induce a classifier to discriminate between interesting and uninteresting items for the user
> - **Hybrid filtering techniques** – combine one or more of the techniques mentioned above

## challenges
> ([[iml/slides/T11_2024_students_wide.pdf#page=30&selection=2,0,60,40|T11_2024_students_wide, p.30]])
> - cold-start problem – concerns the issue that the system cannot draw any inferences for users or items about which it has not yet gathered sufficient information
> - latency problem – new items incorporated into a Rec cannot be used in CF recommendations before a substantial amount of users have evaluated it
> - sparsity problem – few users have rated the same items
> - gray-sheep problem – refers to a user that fall on a border between existing cliques of users

## collaborative filtering systems
- could be using groups, and finding **missing items** based on *users*, or *missing users* based on **items** 
## content-based systems
- closely related to information retrieval
- find items that are similar to items that the user likes
### advantages
- user independence - *no cold start problem*, *no sparsity problem*
- *no grey sheep problem* - can recommend to users with unique preferences
- can recommend *new and unpopular items*
- *transparent*
### disadvantages
- **requires content analysis**
- **portfolio effect** - user is given a homogenous set of alternative
