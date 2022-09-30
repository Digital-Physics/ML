import pandas as pd
from collections import defaultdict

print("we are going to make a Naive Bayes classifier")
print("we are making a multinomial model because the conditional probabilities p(x|y)=0.5 have multiple values")
print("note: we are only choosing one possible bucket/tag per example")
print("we will avoid 0 probabilities by using Laplace smoothing)")
print("on the 'naive' aspects of 'Naive Bayes':")
print("it may be 'naive' to assume all inputs in your data are independent of each other, but it's a simplifying assumption...")
print("that allows joint probabilities to be created by simply multiplying conditional (and prior) probabilities together...")
print("essentially, the presence or absence of a word is considered independent of all the other words in the article")
print("if we didn't do this, the conditional probability of a word vector given an outcome p(x|y) would be expensive to calculate...")
print("the trade-off: the word 'arena' and 'political' are treated separately (and implicitly the phrase 'political arena' is ignored)")
print("so our naive model might classify an article with 'political arena' phrase as 'sports' and not 'politics'")
print("if we didn't make this assumption, you would have to look at the conditional chain rule of probabilities")
print("this would involve conditionally taking into account each other word in the vocabulary word vector (w/ or w/o a notion of order)")
print("another point: it may also be 'naive' to implicitly treat all of our inputs the same, and not weight feature importance")
print("Note: We won't use TF-IDF (Term Frequency * Inverse Document Frequency) when generating our training data feature vectors")
print("TF-IDF might help create feature vectors w/ salient words given higher weights, but we want empirical word probs for Bayes, I think")


class MultinomialNB:
    def __init__(self, articles_per_tag: dict) -> None:
        self.articles_per_tag = articles_per_tag
        self.labels = self.articles_per_tag.keys()
        # self.features = self.vocab.keys()
        self.label_counts = {label: len(self.articles_per_tag[label]) for label in self.articles_per_tag.keys()}
        self.prior_prob_y = {}
        self.cond_prob_X_given_y = {}
        self.features_vocab = {}
        self.train()

    def train(self) -> None:
        """create the probabilities will need for Bayes' Theorem calculations"""
        self.create_vocab()
        self.create_prior_prob_dict()  # for labels, p(y="science"), p(y="tech"), p(y="politics")
        self.create_conditional_prob_dict()  # for feature values, p(X=i|y=j)    used in Bayes' formula

    def create_vocab(self) -> None:
        """create a set of all words across all articles"""
        words_temp = []

        for label in self.labels:
            for article in self.articles_per_tag[label]:
                for word in article:
                    words_temp.append(word)

        self.features_vocab = set(words_temp)

    def create_prior_prob_dict(self) -> None:
        """store priors in a dictionary/hash table. key: value = label: prior_probability"""
        # self.prior_prob_y = {label: len(Xy_train[Xy_train["y"] == label]) / len(discrete_Xy_train["y"]) for label in self.labels}
        self.prior_prob_y = {label: len(self.articles_per_tag[label])/sum(self.label_counts.values()) for label in self.labels}

    def create_conditional_prob_dict(self) -> None:
        """create a 3-layer nested dictionary to hold our conditional probabilities e.g. p(x_7="theory"|y="science")
        print("{feature_col: feature_vals: label_vals: conditional_prob}")
        in the case of words in many articles under one label, aggregate the words and treat like one article."""
        self.cond_prob_X_given_y = {}

        for label in self.labels:
            counter_dict = {}
            for article in self.articles_per_tag[label]:
                for word in article:
                    if word in counter_dict:
                        counter_dict[word] += 1
                    else:
                        counter_dict[word] = 1

        total_words = sum(counter_dict.values())

        for word in self.features_vocab:
            self.cond_prob_X_given_y[]


        # for word in self.features_vocab:
        #     temp_dict = {}
        #     for cat in self.data[feature].unique():
        #         temp_dict2 = {}
        #         for label in self.labels:
        #             # Laplace smoothing adds 1 to numerator and 2 to denominator
        #             numerator_cat_occurences = len(
        #                 discrete_Xy_train[(discrete_Xy_train[feature] == cat) & (discrete_Xy_train["y"] == label)]) + 1
        #             denominator_total_label = len(discrete_Xy_train[discrete_Xy_train["y"] == label]) + 2
        #             temp_dict2[label] = numerator_cat_occurences / denominator_total_label
        #         temp_dict[cat] = temp_dict2
        #     self.cond_prob_X_given_y[feature] = temp_dict
        # print("cond prob X given y (feature_col: feature_vals: label_vals: conditional_prob):")
        # print(self.cond_prob_X_given_y)

    def predict(self, article: list[str]) -> float:
        """Use Naive Bayes' p(y|X = X_1, X_2, ... X_n) = p(X|y)*p(y)/p(X)
                = p(X_1|y)*p(X_2|y)*...p(X_n|y)*p(y)/p(X)"""
        likelihoods = {}

        for label in self.labels:
            # we can take the log of the probabilities (and add instead of multiply) because the log is convex
            # the highest probability will now have the least negative number, and therefore argmax will still make the same label choice
            # this will help prevent us from losing significant digits, which can happen when multiplying many numbers < 0 together
            numerator = math.log(self.prior_prob_y[label])
            # since the denominator is constant p(X) across all posteriors, we can ignore it. we'll scale probs to sum to 1.
            for feat_col, x_input in zip(self.features, article):
                if x_input in self.cond_prob_X_given_y[feat_col]:
                    # += log(probs) instead of *= probs
                    numerator += math.log(self.cond_prob_X_given_y[feat_col][x_input][label])
                else:
                    numerator += math.log(0.5)

            likelihoods[label] = numerator

        print("posterior likelihoods before normalizing to sum to 1")
        print(likelihoods)
        norm_divisor = sum(likelihoods.values())
        for label_key in likelihoods.keys():
            likelihoods[label_key] = likelihoods[label_key] / norm_divisor

        print("after weighting likelihoods to sum to 1")
        print(likelihoods)
        return likelihoods




# is there a better way to get the dictionary to not paste all on one line in PyCharm? (or another IDE)
# "data" dictionary was recreated from the following script which had access to 'articles_per_tag' dictionary:

# print("{")
# for tag_key in articles_per_tag:
#     print("'", tag_key, "': [", sep="")
#     for article in articles_per_tag[tag_key]:
#         print("[")
#         word_count = 0
#         temp_line = ""
#         # groups of 12 words per line
#         for word in article:
#             if word_count < 8:
#                 temp_line = temp_line + "'" + word + "'" + ", "
#                 word_count += 1
#             else:
#                 word_count = 0
#                 print(temp_line)
#                 temp_line = ""
#         print(temp_line)
#         print("],")
#     print("],")
# print("}")


data = {
    'politics': [
        [
            'article', 'writes', 'Joel', 'Furr', 'writes', 'many', 'Mutlus', 'dance',
            'That', 'reminds', 'Armenian', 'massacre', 'Turks', 'Joel', 'took', 'sure',
            'invoke', 'name', 'greps', 'Mason', 'Kibos', 'last', 'name', 'lest',
            'daily', 'rounds', 'dunno', 'Warren', 'Just', 'other', 'heard', 'rumor',
            'Serdar', 'Argic', 'Hasan', 'Mutlu', 'Ahmed', 'Cosar', 'ZUMABOT', 'really',
            'fact', 'Armenian', 'attempting', 'make', 'discussion', 'massacres', 'Armenia', 'Turks',
            'make', 'serious', 'discussion', 'impossible', 'thereby', 'cloaking', 'historical', 'record',
            'tremendous', 'cloud', 'confusion',
        ],
        [
            'Distribution', 'world', 'following', 'posted', 'doubt', 'retyped', 'Yigal', 'Ahrens',
            'importance', 'issue', 'almost', 'total', 'blackout', 'except', 'California', 'reposting',
            'appropriates', 'groups', 'From', 'Times', 'Friday', 'April', '1993', 'EVIDENCE',
            'SEIZED', 'POLICE', 'Richard', 'Paddock', 'Times', 'staff', 'writer', 'FRANCISCO',
            'Thursday', 'served', 'search', 'warrants', 'AntiDefamation', 'League', 'here', 'Angeles',
            'evidence', 'nationwide', 'intelligence', 'network', 'accused', 'keeping', 'files', 'more',
            'political', 'groups', 'newspapers', 'labor', 'unions', 'many', '12000', 'people',
            'operation', 'great', 'detail', 'Francisco', 'authorities', 'simultaneously', 'released', 'voluminous',
            'telling', 'operatives', 'AntiDefamation', 'League', 'searched', 'through', 'trash', 'infiltrated',
            'gather', 'intelligence', 'ArabAmerican', 'rightwing', 'what', 'they', 'called', 'pinko',
            'AntiDefamation', 'League', 'wellknown', 'organization', 'Jewish', 'community', 'dedicated', 'fighting',
            'declined', 'detailed', 'comment', 'Thursday', 'denied', 'breaking', 'laws', 'Police',
            'that', 'organization', 'maintains', 'undercover', 'operatives', 'gather', 'political', 'intelligence',
            'seven', 'cities', 'including', 'Angeles', 'Francisco', 'Groups', 'that', 'were',
            'operation', 'span', 'political', 'spectrum', 'including', 'such', 'groups', 'Klux',
            'White', 'Aryan', 'Resistance', 'Operation', 'Rescue', 'Greenpeace', 'National', 'Assn',
            'Colored', 'People', 'United', 'Farm', 'Workers', 'Jewish', 'Defense', 'League',
            'list', 'were', 'Mills', 'College', 'board', 'directors', 'Francisco', 'public',
            'station', 'KQED', 'Francisco', 'Guardian', 'newspaper', 'People', 'were', 'subjects',
            'included', 'former', 'Republican', 'Pete', 'McCloskey', 'jailed', 'political', 'extremist',
            'LaRouche', 'Angeles', 'Times', 'foreign', 'correspondent', 'Scott', 'Kraft', 'based',
            'Africa', 'Authorities', 'said', 'much', 'material', 'collected', 'groups', 'confidential',
            'obtained', 'illegally', 'from', 'enforcement', 'agencies', 'They', 'also', 'alleged',
            'data', 'some', 'individuals', 'organizations', 'sold', 'separately', 'South', 'African',
            'addition', 'allegations', 'obtaining', 'confidential', 'information', 'from', 'police', 'AntiDefamation',
            'could', 'face', 'total', 'felony', 'counts', 'properly', 'reporting', 'employment',
            'West', 'Coast', 'Bullock', 'according', 'affidavit', 'filed', 'justify', 'search',
            'AntiDefamation', 'League', 'disguised', 'payments', 'Bullock', 'more', 'than', 'years',
            'week', 'Beverly', 'Hills', 'attorney', 'Bruce', 'Hochman', 'then', 'paid',
            'according', 'documents', 'released', 'Francisco', 'Hochman', 'former', 'president', 'Jewish',
            'Council', 'Greater', 'Angeles', 'states', 'leading', 'attorneys', 'will', 'city',
            'late', 'next', 'week', 'could', 'reached', 'comment', 'office', 'said',
            '1990', 'Hochman', 'former', 'prosecutor', 'also', 'member', 'panel', 'appointed',
            'Pete', 'Wilson', 'secretly', 'make', 'initial', 'federal', 'judges', 'California',
            'former', 'regional', 'president', 'AntiDefamation', 'League', 'league', 'which', 'initially',
            'with', 'police', 'denied', 'repeatedly', 'that', 'operation', 'broke', 'laws',
            'officials', 'will', 'confirm', 'deny', 'whether', 'Bullock', 'employee', 'have',
            'they', 'simply', 'traded', 'information', 'with', 'police', 'departments', 'about',
            'might', 'involved', 'hate', 'crimes', 'affidavit', 'filed', 'obtain', 'warrants',
            'searches', 'Francisco', 'police', 'alleged', 'that', 'employees', 'were', 'apparently',
            'than', 'truthful', 'providing', 'information', 'during', 'earlier', 'search', 'conducted',
            'warrant', 'David', 'Lehrer', 'executive', 'director', 'Angeles', 'office', 'said',
            'violated', 'There', 'nothing', 'nefarious', 'about', 'operate', 'what', 'have',
            'said', 'record', 'speaks', 'itself', 'police', 'affidavit', 'contends', 'that',
            'sole', 'control', 'secret', 'fund', 'used', 'factfinding', 'operations', 'Lehrer',
            'documents', 'signed', 'checks', 'from', 'account', 'under', 'name', 'Patterson',
            'said', 'account', 'used', 'subscriptions', 'wide', 'variety', 'extremist', 'publications',
            'might', 'balk', 'sending', 'them', 'directly', 'AntiDefamation', 'League', 'Bullock',
            'collecting', 'intelligence', 'nearly', 'years', 'defended', 'efforts', 'during', 'lengthy',
            'with', 'Francisco', 'police', 'said', 'that', 'gathered', 'names', 'from',
            'sources', 'entered', 'them', 'into', 'computer', 'under', 'headings', 'such',
            'Pinkos', 'that', 'necessarily', 'mean', 'that', 'they', 'were', 'under',
            'might', 'never', 'call', 'them', 'again', 'Bullock', 'said', 'doesnt',
            'anything', 'that', 'theyre', 'files', 'threat', 'anyones', 'civil', 'rights',
            'name', 'appears', 'files', 'under', 'Pinko', 'recent', 'years', 'Bullock',
            'closely', 'with', 'Francisco', 'Police', 'Officer', 'Gerard', 'fled', 'Phillippines',
            'fall', 'after', 'questioned', 'case', 'former', 'employee', 'Gerard', 'supplied',
            'with', 'criminal', 'records', 'Department', 'Motor', 'Vehicles', 'information', 'such',
            'addresses', 'vehicle', 'registration', 'physical', 'drivers', 'license', 'photographs', 'Using',
            'gathered', 'AntiDefamation', 'League', 'Gerard', 'Bullock', 'also', 'provided', 'information',
            'African', 'government', 'receiving', '16000', 'over', 'four', 'years', 'documents',
            'file', 'Times', 'staff', 'writer', 'Kraft', 'which', 'apparently', 'sold',
            'African', 'government', 'provides', 'some', 'insight', 'into', 'hitandmiss', 'nature',
            'file', 'notes', 'that', 'Krafts', 'articles', 'appear', 'frequently', 'Times',
            'researched', 'written', 'little', 'else', 'about', 'file', 'accurate', 'brief',
            'confuses', 'Times', 'Kraft', 'with', 'another', 'Scott', 'Kraft', 'provides',
            'African', 'government', 'with', 'wrong', 'Krafts', 'physical', 'description', 'photograph',
            'personal', 'information', 'Nevertheless', 'documents', 'provide', 'illuminating', 'details', 'Bullock',
            'infiltrated', 'manner', 'organizations', 'from', 'skinheads', 'leftwing', 'radicals', 'searching',
            'through', 'trash', 'target', 'groups', 'Using', 'AntiDefamation', 'League', 'funds',
            'paid', 'informants', 'under', 'code', 'names', 'such', 'Scott', 'Scumbag',
            'closely', 'with', 'police', 'officers', 'down', 'coast', 'exchanged', 'information',
            'worked', 'with', 'federal', 'agencies', 'including', 'Bureau', 'Alcohol', 'Tobacco',
            'Bullocks', 'work', 'paid', 'informant', 'while', 'spying', 'behalf', 'AntiDefamation',
            'South', 'African', 'government', 'that', 'proved', 'undoing', 'learned', 'that',
            'foreign', 'government', 'began', 'investigating', 'leading', 'probe', 'AntiDefamation', 'Leagues',
            'network', 'AntiDefamation', 'League', 'employed', 'undercover', 'operatives', 'gather', 'information',
            'Francisco', 'York', 'Washington', 'Chicago', 'Louis', 'Atlanta', 'according', 'affidavit',
            'Joining', 'Francisco', 'police', 'searching', 'league', 'offices', 'Angeles', 'bank',
            'investigators', 'from', 'office', 'Francisco', 'Dist', 'Atty', 'Arlo', 'Smith',
            'Franchise', 'Board', 'Angeles', 'Police', 'Department', 'which', 'earlier', 'refused',
            'with', 'investigation', 'informed', 'searches', 'Angeles', 'invited', 'participate', 'Investigators',
            'that', 'some', 'confidential', 'information', 'AntiDefamation', 'League', 'files', 'have',
            'from', 'Angeles', 'police', 'officers', 'From', 'Angeles', 'Times', 'Saturday',
            '1993', 'VOWS', 'COOPERATE', 'WITH', 'INVESTIGATION', 'Richard', 'Paddock', 'Times',
            'writer', 'FRANCISCO', 'AntiDefamation', 'League', 'defended', 'record', 'civil', 'rights',
            'Friday', 'said', 'will', 'cooperate', 'with', 'authorities', 'investigating', 'whether',
            'collected', 'confidential', 'police', 'information', 'citizens', 'groups', 'Francisco', 'Dist',
            'Arlo', 'Smith', 'said', 'that', 'AntiDefamation', 'League', 'employees', 'involved',
            'gathering', 'could', 'face', 'many', 'felony', 'counts', 'receiving', 'confidential',
            'eavesdropping', 'violations', 'conspiracy', 'Police', 'have', 'accused', 'AntiDefamation', 'League',
            'truthful', 'about', 'spying', 'operations', 'which', 'collected', 'information', 'more',
            '12000', 'individuals', 'political', 'groups', 'across', 'political', 'spectrum', 'Hundreds',
            'documents', 'released', 'prosecutors', 'Thursday', 'show', 'that', 'maintained', 'nationwide',
            'network', 'kept', 'files', 'political', 'figures', 'Even', 'Smith', 'suggested',
            'AntiDefamation', 'League', 'shut', 'down', 'operation', 'prosecutors', 'would', 'take',
            'into', 'account', 'when', 'deciding', 'what', 'charges', 'file', 'statement',
            'Washington', 'National', 'Director', 'Abraham', 'Foxman', 'described', 'Jewish', 'defense',
            'which', 'fought', 'protect', 'minorities', 'from', 'bigotry', 'discrimination', 'years',
            'said', 'organization', 'regarded', 'credible', 'source', 'extremist', 'groups', 'tradition',
            'providing', 'information', 'police', 'journalists', 'academics', 'government', 'officials', 'public',
            'been', 'policy', 'obtain', 'information', 'illegally', 'said', 'Like', 'other',
            'order', 'protect', 'physical', 'safety', 'sources', 'will', 'comment', 'nature',
            'source', 'information', 'Foxman', 'said', 'AntiDefamation', 'League', 'refused', 'acknowledge',
            'longtime', 'employees', 'Bullock', 'anything', 'more', 'than', 'private', 'individual',
            'informant', 'Among', 'documents', 'released', 'prosecutors', 'were', 'detailed', 'statements',
            'funneled', 'weekly', 'payments', 'Bullock', 'through', 'Beverly', 'Hills', 'attorney',
            'Hochman', 'would', 'penetrate', 'organizations', 'needed', 'this', 'arrangement', 'distanced',
            'Hochman', 'told', 'Francisco', 'police', 'investigator', 'Hochman', 'could', 'reached',
            'home', 'office', 'comment', 'Despite', 'AntiDefamation', 'Leagues', 'assertion', 'that',
            'cooperate', 'with', 'authorities', 'Francisco', 'police', 'said', 'group', 'turn',
            'pertinent', 'documents', 'during', 'voluntary', 'search', 'groups', 'offices', 'Angeles',
            'last', 'fall', 'second', 'round', 'searches', 'Thursday', 'this', 'time',
            'search', 'warrants', 'produced', 'vast', 'quantity', 'records', 'primarily', 'dealing',
            'financial', 'transactions', 'Smith', 'said', 'Further', 'searches', 'necessary', 'will',
            'month', 'before', 'charges', 'filed', 'said', 'investigation', 'course', 'will',
            'facts', 'lead', 'district', 'attorney', 'said', 'Yigal', 'Arens', 'USCISI',
            'arensisiedu',
        ],
        [
            'article', 'Thomas', 'Parsli', 'writes', 'Overall', 'Crime', 'rate', 'felljust',
            'that', 'questions', 'When', 'this', 'have', 'relevant', 'numbers', 'Please',
            'this', 'indication', 'dont', 'believe', 'that', 'youre', 'correct', 'when',
            'occured', 'relevant', 'Acquiring', 'weapons', 'Norway', 'almost', 'kinds', 'weapons',
            'must', 'have', 'permit', 'good', 'reason', 'permit', 'would', 'like',
            'handgun', 'would', 'have', 'gunlicence', 'from', 'police', 'member', 'gunclub',
            'objection', 'beyond', 'ones', 'based', 'ideal', 'RKBA', 'that', 'simply',
            'government', 'should', 'that', 'makes', 'guns', 'plaything', 'tool', 'rich',
            'discriminates', 'against', 'poor', 'selfdefense', 'considered', 'appropriate', 'under', 'what',
            'allowed', 'instance', 'protection', 'youre', 'going', 'carrying', 'very', 'large',
            'regular', 'basis', 'have', 'been', 'threatened', 'police', 'would', 'check',
            'records', 'SERIOUS', 'crimes', 'andor', 'records', 'SERIOUS', 'mental', 'diseases',
            'been', 'suggested', 'generally', 'supported', 'among', 'owners', 'What', 'many',
            'that', 'many', 'most', 'proposals', 'contain', 'sort', 'gotcha', 'clause',
            'allows', 'arbitrary', 'denial', 'even', 'qualify', 'every', 'licence', 'would',
            'active', 'member', 'club', 'months', 'BEFORE', 'could', 'collect', 'little',
            'getting', 'drivers', 'licence', 'isnt', 'have', 'prove', 'that', 'drive',
            'allowed', 'this', 'point', 'should', 'pointed', 'that', 'general', 'drivers',
            'most', 'part', 'nothing', 'like', 'European', 'counterpart', 'understand', 'getting',
            'difficult', 'there', 'than', 'here', 'joke', 'usual', 'objection', 'that',
            'discussing', 'different', 'things', 'instance', 'drivers', 'license', 'permit', 'operate',
            'vehicle', 'public', 'road', 'necessary', 'operate', 'private', 'property', 'That',
            'require', 'driving', 'permits', 'generally', 'considered', 'arise', 'from', 'governments',
            'power', 'enact', 'reasonable', 'regulations', 'behavior', 'public', 'lands', 'permit',
            'instance', 'which', 'closer', 'analogy', 'would', 'much', 'harder', 'thing',
            'legally', 'since', 'wouldnt', 'based', 'making', 'regulations', 'public', 'property',
            'activity', 'private', 'property', 'guns', 'crimes', 'Norway', 'Some', 'crimes',
            'with', 'guns', 'that', 'have', 'been', 'owners', 'arms', 'long',
            'these', 'rather', 'exeption', 'Most', 'criminals', 'accuire', 'guns', 'them',
            'mostly', 'short', 'time', 'befor', 'crime', 'knives', 'allowed', 'cary',
            'public', 'your', 'belt', 'open', 'Americans', 'think', 'have', 'carry',
            'public', 'rigth', 'This', 'varies', 'widely', 'thing', 'think', 'Europeans',
            'difficult', 'time', 'with', 'that', 'fifty', 'unique', 'jurisdictions', 'where',
            'from', 'state', 'another', 'radically', 'different', 'from', 'country', 'Europe',
            'Some', 'places', 'allow', 'open', 'carry', 'both', 'guns', 'knives',
            'allow', 'concealed', 'Some', 'prohibit', 'both', 'allow', 'other', 'either',
            'local', 'restriciton', 'Individual', 'masses', 'individual', 'more', 'important', 'than',
            'only', 'some', 'extent', 'Your', 'criminal', 'laws', 'protect', 'individuals',
            'masses', 'What', 'happens', 'when', 'rigths', 'some', 'individuals', 'affects',
            'others', 'question', 'must', 'asked', 'right', 'this', 'individual', 'affecting',
            'this', 'other', 'individual', 'What', 'usually', 'that', 'rights', 'this',
            'meaning', 'some', 'individuals', 'within', 'this', 'group', 'here', 'defined',
            'guns', 'adversely', 'affecting', 'rights', 'some', 'other', 'group', 'instance',
            'using', 'attack', 'Steve', 'youd', 'have', 'point', 'essentially', 'what',
            'discussing', 'that', 'becuase', 'some', 'person', 'qualifies', 'member', 'group',
            'guns', 'then', 'some', 'third', 'person', 'perhaps', 'another', 'time',
            'told', 'that', 'their', 'being', 'member', 'that', 'group', 'taking',
            'somebody', 'elses', 'rights', 'like', 'trying', 'punish', 'newspapers', 'libel',
            'issue', 'believe', 'issue', 'GUNS', 'gunlegislation', 'issue', 'crime', 'violence',
            'question', 'what', 'extent', 'guns', 'legislation', 'impact', 'those', 'shouldnt',
            'items', 'that', 'serve', 'lived', 'Amerika', 'would', 'probably', 'have',
            'myselfe', 'HOME', 'should', 'have', 'like', 'that', 'course', 'would',
            'didnt', 'have', 'fear', 'that', 'other', 'people', 'might', 'into',
            'twisted', 'little', 'minds', 'hurt', 'currently', 'dont', 'have', 'that',
            'expect', 'will', 'think', 'wise', 'sell', 'guns', 'like', 'candy',
            'states', 'state', 'does', 'case', 'theres', 'limit', 'which', 'state',
            'wisdom', 'Freedom', 'general', 'unwise', 'concept', 'preemptively', 'restrict', 'everything',
            'might', 'unwise', 'then', 'freedom', 'becomes', 'meaningless', 'concept', 'believe',
            'have', 'driverslicence', 'think', 'should', 'free', 'guns', 'raise', 'hand',
            'drivers', 'licenses', 'currently', 'implemented', 'theyre', 'waste', 'time', 'little',
            'than', 'revanue', 'generation', 'State', 'ignored', 'startling', 'number', 'drivers',
            'guarantee', 'level', 'skill', 'higher', 'than', 'necessary', 'your', 'road',
            'somebody', 'else', 'killed', 'knowledge', 'traffic', 'laws', 'beyond', 'what',
            'will', 'have', 'picked', 'riding', 'around', 'parents', 'mentioned', 'theyre',
            'things', 'David', 'Veal', 'Univ', 'Tenn', 'Cont', 'Education', 'Info',
            'Group', 'still', 'remember', 'laughed', 'your', 'pushed', 'down', 'elevator',
            'beginning', 'think', 'dont', 'love', 'anymore', 'Weird',
        ],
        [
            'Marc', 'Afifi', 'writes', 'Lets', 'forget', 'that', 'soldiers', 'were',
            'murdered', 'distinction', 'trivial', 'Murder', 'happens', 'innocent', 'people', 'people',
            'line', 'work', 'kill', 'killed', 'just', 'happened', 'that', 'these',
            'line', 'duty', 'were', 'killed', 'opposition', 'That', 'still', 'doesnt',
            'should', 'cheer', 'their', 'deaths', 'Policemen', 'also', 'line', 'fire',
            'includes', 'possibility', 'getting', 'killed', 'Should', 'happy', 'when', 'they',
            'before', 'question', 'whether', 'agree', 'with', 'policies', 'Israel', 'wish',
            'cease', 'occupation', 'dont', 'rejoice', 'death', 'marc',
        ],
        [
            'Thomas', 'Farrell', 'writes', 'feel', 'that', 'defendents', 'should', 'have',
            'convicted', 'regardless', 'evidence', 'that', 'would', 'truely', 'civil', 'rights',
            'know', 'about', 'everybody', 'else', 'they', 'should', 'have', 'been',
            'BECAUSE', 'evidence', 'which', 'mind', 'quite', 'sufficient', 'court', 'room',
            'case', 'After', 'careful', 'consideration', 'have', 'come', 'your', 'conclusion',
            'good',
        ],
        [
            'minor', 'point', 'interest', 'earlier', 'news', 'reports', 'claim', 'have',
            'quoting', 'Governor', 'Texas', 'when', 'Holiness', 'referred', 'Dividians', '_Mormons_',
            'their', 'expulsion', 'from', 'Texans', 'have', 'details',
        ],
        [
            'Distribution', 'world', 'article', 'Steve', 'Manes', 'writes', 'Morris', 'wrote',
            'Neal', 'Knox', 'Firearms', 'Coalition', 'points', 'full', 'force', 'antigun',
            'class', 'their', 'multimillions', 'their', 'polling', 'organizations', 'their', 'schools',
            'news', 'media', 'their', 'entertainment', 'media', 'entertainment', 'media', 'force',
            'ruling', 'class', 'this', 'same', 'media', 'thats', 'made', 'billions',
            'films', 'television', 'that', 'glorify', 'guns', 'users', 'that', 'another',
            'media', 'Youve', 'kidding', 'this', 'mean', 'that', 'consider', 'absolutely',
            'media', 'guilty', 'hypocrisy', 'Note', 'that', 'film', 'industry', 'California',
            'their', 'political', 'support', 'assault', 'weapon', 'state', 'amendment', 'bill',
            'entertainment', 'industry', 'from', 'that', 'very', 'Note', 'that', 'very',
            'Batman', 'comic', 'book', 'Seduction', 'that', 'produced', 'tool', 'guncontrol',
            'carries', 'back', 'page', 'Terminator', 'video', 'game', 'extolling', 'numerous',
            'sophisticated', 'weapons', 'available', 'player', 'Note', 'that', 'Arthur', 'Ochs',
            'publisher', 'Times', 'oldest', 'most', 'incessant', 'guncontrol', 'grinders', 'himself',
            'concealed', 'handgun', 'Still', 'find', 'completely', 'incredible', 'that', 'these',
            'live', 'aphorism', 'believe', 'that', 'speak', 'company', 'write', 'today',
            'Investors', 'Packet',
        ],
        [
            'believe', 'that', 'individuals', 'should', 'have', 'right', 'weapons', 'mass',
            'find', 'hard', 'believe', 'that', 'would', 'support', 'neighbors', 'right',
            'nuclear', 'weapons', 'biological', 'weapons', 'nerve', 'hisher', 'property', 'There',
            'having', 'biological', 'weapons', 'nerve', 'hisher', 'property', 'even', 'walking',
            'property', 'with', 'such', 'items', 'ipso', 'facto', 'ones', '_RIGHT_',
            'such', 'weapons', 'mass', 'destruction', 'Hell', 'patent', 'office', 'patents',
            'nerve', 'that', 'anyone', 'obtain', 'simply', 'sending', 'Patent', 'Office',
            'These', 'same', 'patents', 'verboten', 'English', 'citizens', 'from', 'their',
            'office', 'which', 'doesnt', 'surprise', 'based', 'mistrust', 'government', 'against',
            'ownership', 'semiautomatic', 'rifles', 'cannot', 'even', 'agree', 'keeping', 'weapons',
            'destruction', 'hands', 'individuals', 'there', 'hope', 'saying', 'should', 'have',
            'prohibiting', 'owning', 'biological', 'warfare', 'agents', 'nerve', 'agents', 'Will',
            'laws', 'against', 'owning', 'chlorine', 'cyanide', 'well', 'Will', 'pass',
            'against', 'owning', 'acetylene', 'that', 'could', 'have', 'been', 'used',
            'Bradley', 'IFVs', 'Branch', 'Dividians', 'known', 'their', 'anticombustion', 'engine',
            'Will', 'pass', 'laws', 'against', 'owning', '5gallon', 'cylinders', 'propane',
            'they', 'could', 'have', 'been', 'used', 'flame', 'throwers', 'proverbial',
            'Hell', 'always', 'Good',
        ],
        [
            'think', 'unlikely', 'that', 'Clinton', 'policy', 'wonk', 'facilitators', 'arranged',
            'raid', 'display', 'piece', 'Constitution', 'Look', 'what', 'Bush', 'administration',
            'Drug', 'that', 'baggie', 'crack', 'George', 'waved', 'cameras', 'They',
            'dealer', 'from', 'ghetto', 'brought', 'White', 'House', 'they', 'could',
            'been', 'dealt', 'White', 'House', 'Lawn', 'dont', 'think', 'anybody',
            'honestly', 'think', 'Clinton', 'would', 'have', 'moral', 'qualms', 'about',
            'only', 'really', 'worrisome', 'thing', 'that', 'heroic', 'defense', 'their',
            'will', 'make', 'Clintons', 'Constitution', '_more_', 'wanted', 'media', 'politicians',
            'filter', 'this', 'that', 'general', 'public', 'will', 'think', 'guys',
            'help', 'them', 'Stand', 'with', 'your', 'friends', 'family', 'adnd',
            'anytime', 'cantheir', 'supposed', 'moral', 'qualms', 'important', 'issue', 'They',
            'fight', 'against', 'oppressive', 'government', 'could', 'just', 'well', 'have',
            'Brian', 'Watkins',
        ],
        [
            'Sessions', 'writes', 'Steve', 'Lets', 'here', 'what', 'zionism', 'Assuming',
            'mean', 'hear', 'werent', 'listening', 'just', 'told', 'Zionism', 'Racism',
            'tautological', 'statement', 'think', 'confusing', 'tautological', 'with', 'false', 'misleading',
            'Stein',
        ],
        [
            'eyore', 'heading', 'true', 'Frank', 'should', 'ashamed', 'himself', 'Nothing',
            'more', 'than', 'people', 'dont', 'respect', 'rights', 'others', 'voice',
            'opinions', 'idol', 'Lenny', 'Bruce', 'once', 'commented', 'about', 'that',
            'Time', 'Magazine', 'when', 'they', 'advocated', 'censorship', 'material', 'Time',
            'sided', 'with', 'cops', 'their', 'arresting', 'Bruce', 'shows', 'whereby',
            'would', 'cocksucker', 'then', 'cops', 'would', 'rush', 'stage', 'arrest',
            'havent', 'changed', 'cant', 'help', 'think', 'Lenny', 'would', 'received',
            'politically', 'correct', 'arena', 'Heck', 'even', 'support', 'right', 'nazis',
            'their', 'opinions', 'march', 'down', 'streets', 'before', 'Frank', 'anyone',
            'makes', 'wisecracks', 'about', 'antiSemitismIm', 'Jewish', 'longtime', 'member', 'AIPAC',
            'contributed', 'over', '1000', 'apiece', 'these', 'fine', 'groups', 'regular',
            'every', 'proIsrael', 'group', 'find', 'still', 'support', 'right', 'people',
            'speak', 'vomit', 'propaganda', 'want', 'know', 'just', 'these', 'people',
            'this', 'assumption', 'that', 'Frank', 'indeed', 'write', 'some', 'sysadmin',
            'Teel', 'admonished', 'this', 'case', 'hereby', 'retract', 'these', 'nasties',
            'toward', 'stand', 'against', 'Frank', 'trashing', 'First', 'Amendment', 'Philip',

        ],
        [
            'article', 'Tavares', 'writes', 'article', 'Foxvog', 'Douglas', 'writes', 'article',
            'writes', 'article', 'John', 'Lawrence', 'Rutledge', 'writes', 'massive', 'destructive',
            'many', 'modern', 'weapons', 'makes', 'cost', 'accidental', 'crimial', 'usage',
            'weapons', 'great', 'weapons', 'mass', 'destruction', 'need', 'control', 'government',
            'Individual', 'access', 'would', 'result', 'needless', 'deaths', 'millions', 'This',
            'right', 'people', 'keep', 'bear', 'many', 'modern', 'weapons', 'nonexistant',
            'stating', 'where', 'youre', 'coming', 'from', 'Needless', 'disagree', 'every',
            'believe', 'that', 'individuals', 'should', 'have', 'right', 'weapons', 'mass',
            'find', 'hard', 'believe', 'that', 'would', 'support', 'neighbors', 'right',
            'nuclear', 'weapons', 'biological', 'weapons', 'nerve', 'hisher', 'property', 'cannot',
            'agree', 'keeping', 'weapons', 'mass', 'destruction', 'hands', 'individuals', 'there',
            'dont', 'sign', 'blank', 'checks', 'course', 'term', 'must', 'rigidly',
            'bill', 'When', 'Doug', 'Foxvog', 'says', 'weapons', 'mass', 'destruction',
            'nukes', 'When', 'Sarah', 'Brady', 'says', 'weapons', 'mass', 'destruction',
            'Street', 'Sweeper', 'shotguns', 'semiautomatic', 'rifles', 'doubt', 'uses', 'this',
            'that', 'using', 'quote', 'allegedly', 'from', 'back', 'When', 'John',
            'Rutledge', 'says', 'weapons', 'mass', 'destruction', 'then', 'immediately', 'follows',
            'thousands', 'people', 'killed', 'each', 'year', 'handguns', 'this', 'number',
            'reduced', 'putting', 'reasonable', 'restrictions', 'them', 'what', 'does', 'Rutledge',
            'term', 'read', 'article', 'presenting', 'first', 'argument', 'about', 'weapons',
            'destruction', 'commonly', 'understood', 'then', 'switching', 'other', 'topics', 'first',
            'evidently', 'show', 'that', 'weapons', 'should', 'allowed', 'then', 'later',
            'given', 'this', 'understanding', 'consider', 'another', 'class', 'believe', 'that',
            'company', 'write', 'today', 'special', 'Investors', 'Packet', 'doug', 'foxvog',
        ],
        [
            'Would', 'asking', 'much', 'DOCUMENT', 'these', 'allegations', 'Israel', 'used',
            'kill', 'neutral', 'reporters', 'think', 'confuse', 'Israel', 'with', 'other',
            'that', 'geographical', 'region', 'which', 'notion', 'free', 'unmonitored', 'government',
            'corps', 'would', 'joke', 'notion', 'that', 'Israel', 'threatens', 'human',
            'Palestinians', 'sealing', 'Gaza', 'strip', 'real', 'When', 'civil', 'stops',
            'behave', 'like', 'mature', 'human', 'beings', 'Israel', 'will', 'talk',
            'both', 'sides', 'peace', 'before',
        ],
        [
            'mahogany126', 'Organization', '1991', 'World', 'Champion', 'Minnesota', 'Twins', 'Distribution',
            'paulhshcom', 'Paul', 'Havemann', 'writes', 'article', 'Russ', 'Anderson', 'writes',
            'Mark', 'Wilson', 'writes', 'This', 'past', 'Thursday', 'GOre', 'threw',
            'ball', 'home', 'opener', 'Atlanta', 'Braves', 'According', 'news', 'reports',
            'loudly', 'booed', 'Norman', 'these', 'were', 'your', 'typical', 'beer',
            'rednecks', 'Personally', 'wouldnt', 'have', 'paid', 'more', 'attention', 'incident',
            'that', 'evening', 'news', 'when', 'describing', 'event', 'went', 'comment',
            'being', 'booed', 'nothing', 'unusual', 'since', 'normal', 'audiences', 'this',
            'since', 'celebrity', 'delaying', 'start', 'game', 'What', 'bunch', 'crock',
            'never', 'heard', 'incident', 'which', 'thrower', 'ceremonial', 'ball', 'been',
            'before', 'Quayle', 'roundly', 'booed', 'Milwaulkee', 'last', 'year', 'listening',
            'This', 'game', 'that', 'Quayle', 'told', 'Brewers', 'players', 'that',
            'like', 'them', 'play', 'Orioles', 'ALCS', 'come', 'this', 'Defending',
            'comparing', 'Quayle', 'compared', 'Quayle', 'Gore', 'Mark', 'said', 'never',
            'incident', 'which', 'thrower', 'ceremonial', 'ball', 'been', 'booed', 'before',
            'another', 'incident', 'media', 'liberal', 'bias', 'sure', 'would', 'have',
            'Quayle', 'incident', 'compare', 'Quayle', 'anyone', 'most', 'likely', 'would',
            'Fudd', 'that', 'about', 'says', 'back', 'with', 'back', 'altfan',
            'Begone', 'Russ', 'Anderson', 'Disclaimer', 'statements', 'reflect', 'upon', 'employer',
            'else', '1993', 'EXTwins', 'Jack', 'Morris', 'innings', 'pitched', 'runs',
            'Series',
        ],
        [
            'words', 'chilling', 'effect', 'stimulate', 'impulses', 'within', 'that', 'small',
            'neurons', 'call', 'brain', 'been', 'days', 'know', 'where', 'your',
            'Slick', 'Willys', 'already', 'hand', 'pocket', 'just', 'afraid', 'what',
            'grab', 'hold',
        ],
    ],
    'sports': [
        [
            'article', 'writes', 'just', 'wanted', 'everyone', 'know', 'that', 'have',
            'what', 'little', 'respect', 'have', 'LeFebvre', 'after', 'seeing', 'todays',
            'game', 'dishard', 'think', 'thats', 'just', 'wait', 'until', 'tries',
            'leadoff', 'spot', 'again', 'also', 'wonder', 'they', 'with', 'this',
            'never', 'believed', 'managers', 'that', 'much', 'with', 'winning', 'until',
            'they', 'with', 'losing', 'Rick',
        ],
        [
            'Phillies', 'salvaged', 'their', 'weekend', 'series', 'against', 'Chicago', 'Cubs',
            'them', '1110', 'wild', 'Wrigley', 'Field', 'Sunday', 'afternoon', 'Phils',
            'three', 'game', 'series', 'first', 'time', 'Phillies', 'have', 'lost',
            'young', 'season', 'Phils', 'jumped', 'lead', 'game', 'thanks', 'John',
            '2run', 'homers', 'Chamberlain', 'homers', 'However', 'Danny', 'Jackson', 'Phillies',
            'relief', 'unable', 'hold', 'lead', 'Mitch', 'Williams', 'entered', 'game',
            'Phillies', 'leading', 'however', 'Candy', 'Maldonado', 'ninth', 'inning', 'homerun',
            'Dave', 'Hollins', 'threerun', 'shot', 'first', 'year', 'push', 'Phils',
            'stay', 'However', 'shaky', 'bottom', '11th', 'Cubs', 'scored', 'runs',
            'runner', 'base', 'when', 'Cubs', 'pinch', 'Randy', 'Myers', 'Scanlan',
            'were', 'position', 'players', 'Myers', 'bunted', 'into', 'double', 'play',
            'Phils', 'bring', 'their', 'league', 'leading', 'record', 'back', 'action',
            'Wednesday', 'Thursday', 'against', 'Padres',
        ],
        [
            'people', 'here', 'stupid', 'what', 'breaker', 'cause', 'they', 'have',
            'same', 'record', 'people', 'sooooo', 'stuppid', 'first', 'list', 'breaker',
            'there', 'different', 'record', 'thought', 'people', 'this', 'good', 'with',
            'might', 'great', 'Math', 'tell', 'teams', 'ahve', 'same', 'points',
            'different', 'record', 'Manretard', 'Cant', 'believe', 'people', 'actually', 'first',

        ],
        [
            'article', 'writes', 'BEAT', 'PITTSBURGH', 'IIIKevin', 'Stevens', 'AFighting', '1Call',
            '2Call', 'Domi', '3Call', 'grandmother', 'Shed', 'kick', 'YeahIve', 'seen',
            'grand', 'motherI', 'could', 'Joseph', 'Stiehm',
        ],
        [
            'Organization', 'Nokia', 'article', 'Richard', 'John', 'Rauser', 'writes', 'Heres',
            'there', 'many', 'Europeans', 'sick', 'watching', 'game', 'between', 'American',
            'team', 'lets', 'Wings', 'Canucks', 'seeing', 'names', 'like', 'Bure',
            'Borshevshky', 'this', 'North', 'America', 'isnt', 'Toronto', 'Detriot', 'Quebec',
            'particularly', 'annoying', 'numbers', 'Euros', 'other', 'teams', 'getting', 'worse',
            'sick', 'watching', 'allamerican', 'names', 'like', 'GRETZKY', 'Which', 'names',
            'Sitting', 'bull', 'dances', 'with', 'wolves', 'North', 'America', 'What',
            'here', 'Jyri',
        ],
        [
            'stephcsuiucedu', 'Dale', 'Stephenson', 'writes', 'Compiled', 'from', 'last', 'five',
            'Average', 'reports', 'here', 'career', 'individual', 'players', 'reports', 'Stats',
            'Sherri', 'Nichols', 'Players', 'listed', 'descending', 'order', 'some', 'comments',
            'some', 'players', 'deleted', 'Third', 'Basemen', 'Name', '1988', '1989',
            '1991', '1992', '8892', 'Mitchell', 'Kevin', '0690', 'that', 'Kevin',
            'never', 'would', 'have', 'expected', 'spot', 'Gonzales', 'Rene', '0685',
            'that', 'first', 'names', '1988', 'only', 'with', 'first', 'second',
            '1988', 'year', 'glove', 'Average', 'points', 'higher', 'both', 'leagues',
            'other', 'year', 'Leius', 'Scott', '0672', 'Looks', 'good', 'moving',
            'Pendleton', 'Terry', '0667', 'Highest', 'fiveyear', 'regular', 'though', 'only',
            'good', 'Kevin', 'Mitchell', 'Ventura', 'Robin', '0657', 'Wallach', '0657',
            'Kelly', '0650', 'other', 'elite', 'fielders', 'league', 'Pagliarulo', 'Mike',
            'This', 'interesting', 'line', '1988', 'figure', 'slightly', 'below', 'average',
            'pathetic', '1991', 'next', 'best', 'year', 'anybody', 'Part', 'that',
            '1988', 'with', 'Yankees', '1990', 'with', 'Padres', 'appear', 'have',
            'infield', '1991', 'with', 'Twins', 'judging', 'Leius', 'Gaetti', 'Metrodome',
            'place', 'play', 'third', 'Williams', 'Matt', '0647', 'another', 'elite',
            'list', 'Caminiti', '0642', 'Sabo', 'Chris', '0642', 'fielders', 'whose',
            'average', 'overstate', 'their', 'value', 'dont', 'know', 'what', 'happened',
            'judging', 'three', 'previous', 'years', '1992', 'fluke', 'Sabo', 'merely',
            'however', 'incredible', '1988', 'best', 'year', 'ever', 'brings', 'average',
            'Steve', '0635', 'Strange', 'last', 'years', 'Schmidt', 'Mike', '0628',
            'reputation', 'best', 'fielders', 'ever', 'third', 'base', 'below', 'average',
            '1988', 'Boggs', 'Wade', '0626', 'Boggs', 'been', 'pretty', 'good',
            'know', 'what', 'happened', '1990', 'every', 'other', 'year', 'been',
            'average', 'usually', 'quite', 'Martinez', 'Egdar', '0624', 'Last', 'year',
            'portent', 'Average', '0619', 'Seitzer', 'Kevin', '0616', 'Average', '0615',
            'leagues', 'usually', 'have', 'defensive', 'averages', 'very', 'close', 'another',
            'different', 'from', 'year', 'year', 'ideas', 'Jacoby', 'Brook', '0613',
            'declining', 'Hansen', 'Dave', '0611', 'Magadan', 'Dave', '0609', 'Jefferies',
            '0606', 'Three', 'firsttime', 'regulars', 'above', 'average', '1992', 'sure',
            'gets', 'grief', 'about', 'fielding', 'never', 'good', 'year', 'while',
            'improved', 'become', 'average', 'fielder', 'average', 'fielder', 'third', 'Zeile',
            '0605', 'Zeile', 'other', 'hand', 'below', 'average', 'fielder', 'Each',
            'about', 'points', 'below', 'average', 'probably', 'just', 'park', 'since',
            'Pendleton', 'excellent', 'three', 'years', 'before', 'this', 'Baerga', 'Carlos',
            'Moving', 'back', 'second', 'good', 'idea', 'Hayes', 'Chris', '0602',
            'supposed', 'good', 'defensively', 'grand', 'total', 'year', 'above', 'league',
            'last', 'year', 'Johnson', 'Howard', '0588', 'Lansford', 'Carney', '0587',
            'Johnson', 'Carney', 'Lansford', 'separated', 'birth', 'credit', 'HoJo', 'have',
            'average', 'year', '1990', 'Lansford', 'couldnt', 'even', 'break', 'mark',
            'help', 'year', 'glove', 'Hollins', 'Dave', '0577', 'Good', 'hitter',
            'needs', 'work', 'Sheffield', 'Gary', '0575', 'good', 'fielder', 'Blauser',
            '0573', 'Fryman', 'Travis', '0571', 'Both', 'better', 'shortstop', 'Gomez',
            'consecutive', 'horrible', 'years', 'Camden', 'Yards', 'doesnt', 'seem', 'have',
            'fielding', 'Palmer', 'Dean', '0520', 'Texas', 'slugger', 'debuts', 'with',
            'lowest', 'career', 'lowest', 'third', 'ever', 'Dean', 'Dale', 'Stephenson',
            'Grad', 'Student', 'Large', 'considered', 'good', 'look', 'wise', 'especially',
            'overburdened', 'with', 'information', 'Golden', 'Kimball',
        ],
        [
            'Weiss', 'played', 'second', 'White', 'early', 'sixties', 'chiefly', 'back',
            'Good', 'glove', 'some', 'spunk', 'Which', 'reminds', 'they', 'still',
            'Kosher', 'dogs', 'Comiskey', 'Mark', 'Bernstein', 'Eastgate', 'Systems', 'Main',
            'Watertown', '02172', 'voice', '5621638', '1617', '9249044', 'Compuserve', '76146262',
        ],
        [
            'Distribution', 'Expires', '5993', 'Summary', 'OPCY', 'just', 'small', 'Hell',
            'Opening', 'game', 'could', 'easily', 'largest', 'history', 'stadium', 'with',
            'seats', 'unfortunely', 'Yards', 'definitely', 'excellent', 'ballpark', 'only', 'holds',
            '45000', 'with', 'spots', 'Ticket', 'sales', 'entire', 'year', 'moving',
            'Bleacher', 'seats', 'almost', 'gone', 'every', 'game', 'this', 'year',
            'likelyhood', 'that', 'could', 'sell', 'every', 'game', 'this', 'year',
            'lead', 'division', 'most', 'year', 'like', 'another', 'front', 'sale',
            'anyone', 'likely', 'forced', 'upon', 'Jacobs', 'major', 'debt', 'apparently',
            'owner', 'willing', 'spend', 'proven', 'rightfielder', 'free', 'agent', 'winter',
            'made', 'fifth', 'starter', 'pitching', 'staff', 'looks', 'pretty', 'good',
            'Mussina', 'McDonald', 'Rhodes', 'Fernando', 'Baltimore', 'pick', 'victors', 'very',
            'East', 'Admiral', 'Steve', 'Internet', 'Address', 'Committee', 'Liberation', 'Intergration',
            'Organisms', 'their', 'Rehabilitation', 'Into', 'Society', 'from', 'Dwarf', 'Polymorph',
            'greatest', 'female', 'rock', 'band', 'that', 'ever', 'existed', 'This',
            'brought', 'Frungy', 'Sport', 'Kings', 'Second', 'last', 'season', 'Gregg',
            'Wild', 'Thing', 'Olson', 'uncorks', 'wild', 'pitch', 'allowing', 'Blue',
            'Blue', 'Jays', '11th', 'ends', 'Baby', 'Birds', 'miracle', 'season',
        ],
        [
            'Would', 'someone', 'please', 'give', 'address', 'Texas', 'Ranger', 'ticket',
            'Thanks', 'very', 'much',
        ],
        [
            'Distribution', 'world', 'Does', 'anybody', 'have', 'Tiger', 'Stadium', 'seating',
            'Thanks', 'Brian', 'Curran', 'Mead', 'Data', 'Central', 'didnt', 'think',
            'been', 'asked', 'catch', 'when', 'temperature', 'below', 'Carlton', 'Fisk',
            'White', 'catcher', 'playing', 'during', '40degree', 'April', 'ball', 'game',
        ],
        [
            'article', 'writes', 'recently', 'been', 'working', 'project', 'determine', 'greatest',
            'their', 'respective', 'postions', 'sources', 'Total', 'Baseball', 'James', 'Historical',
            'Ballplayers', 'biography', 'word', 'mouth', 'biased', 'opinions', 'Feel', 'free',
            'suggest', 'flame', 'whateverbut', 'tried', 'objective', 'possible', 'using', 'statistical',
            'inlcuded', 'timeconviences', 'sake', 'judged', 'Total', 'Average', 'fielding', 'rangeruns',
            'player', 'rating', 'Total', 'Baseball', 'stolen', 'bases', 'curiositys', 'sake',
            'years', 'playedMVP', 'Career', 'Gehrig', 'Jimmie', 'Foxx', 'Eddie', 'Murray',
            'Greenberg', 'Johnny', 'Mize', 'Willie', 'McCovey', 'Dick', 'Allen', 'Harmon',
            'Kieth', 'Hernandez', 'before', 'except', 'after', 'people', 'named', 'kEIth',
            'Terry', 'George', 'Sisler', 'Eddie', 'Collins', 'Morgan', 'Jackie', 'Robinson',
            'Hornsby', 'Lajoie', 'Rhyne', 'Sandberg', 'Learn', 'spell', 'Ryne', 'Charlie',
            'Carew', 'Bobby', 'Grich', 'Bobby', 'Doerr', 'Honus', 'Wagner', 'Ripken',
            'Lloyd', 'Ozzie', 'Smith', 'Robin', 'Yount', 'Cronin', 'Arky', 'Vaughan',
            'Appling', 'Ernie', 'Banks', 'Boudreau', 'Mike', 'Schmidt', 'Matthews', 'George',
            'Wade', 'Boggs', 'Santo', 'Brooks', 'Robinson', 'Frank', 'Baker', 'Darrell',
            'Traynor', 'Dandridge', 'Brooks', 'think', 'would', 'least', 'ahead', 'Santo',
            'Gibson', 'Darren', 'Daulton', '1993', 'Yogi', 'Berra', 'Johnny', 'Bench',
            'Cochrane', 'Bill', 'Dickey', 'Gabby', 'Hartnett', 'Campanella', 'Gary', 'Carter',
            'Fisk', 'Thurman', 'Munson', 'Williams', 'Stan', 'Musial', 'Rickey', 'Henderson',
            'Yastrzemski', 'Barry', 'Bonds', 'Raines', 'Jackson', 'Ralph', 'Kiner', 'Willie',
            'Simmons', 'Willie', 'Mays', 'Cobb', 'Tris', 'Speaker', 'Mickey', 'Mantle',
            'Oscar', 'Charleston', 'Andre', 'Dawson', 'Duke', 'Snider', 'Kirby', 'Puckett',
            'Murphy', 'Babe', 'Ruth', 'Hank', 'Aaron', 'Frank', 'Robinson', 'Kaline',
            'Jackson', 'Dave', 'Winfield', 'Roberto', 'Clemente', 'Tony', 'Gwynn', 'Pete',
            'Walter', 'Johnson', 'Lefty', 'Grove', 'Young', 'Christy', 'Mathewson', 'Pete',
            'Seaver', 'Roger', 'Clemens', 'Gibson', 'Warren', 'Spahn', 'Satchel', 'Paige',
            'Marichal', 'Whitey', 'Ford', 'Feller', 'Palmer', 'Steve', 'Carlton', 'Overall',
            'Ruth', 'Williams', 'Mays', 'Cobb', 'Aaron', 'Wagner', 'Speaker', 'Schmidt',
            'Mantle', 'Musial', 'DiMaggio', 'FRobinson', 'Grove', 'Henderson', 'JGibson', 'CYoung',
            'Foxx', 'Mathewson', 'Alexander', 'Morgan', 'JRobinson', 'Hornsby', 'Seaver', 'Clemens',
            'Lajoie', 'Yastrzemski', 'Kaline', 'Brett', 'Gibson', 'Spahn', 'Charleston', 'Berra',
            'Lloyd', 'Raines', 'Sandberg', 'Gehringer', 'OSmith', 'Yount', 'BaBonds', 'Paige',
            'Marichal', 'Ford', 'Feller', 'Boggs', 'Again', 'feel', 'free', 'comment',

        ],
    ],
    'tech': [
        [
            'Thanks', 'Steve', 'your', 'helpful', 'informative', 'comments', 'stereo', 'sound',
            'developers', 'arent', 'addressing', 'problem', 'This', 'make', 'trusty', 'superior',
            'replaced', 'with', 'though', 'Thanks', 'Doug',
        ],
        [
            'Please', 'unsubscribe', 'This', 'user', 'become', 'inactive', 'wish', 'discontinue',
            'this', 'mailing', 'list', 'Marc', 'Newman',
        ],
        [
            'just', 'read', 'article', 'SWII', 'thing', 'puzzles', 'article', 'says',
            'serialonly', 'device', 'Does', 'that', 'mean', 'have', 'unplug', 'modem',
            'time', 'want', 'print', 'something', 'printer', 'port', 'also', 'serial',
            'interface', 'ImageWriter', 'Kris', 'System', 'fourdcom', 'Phone', '6174940565', 'Cute',
            'Being', 'computer', 'means', 'never', 'having', 'youre', 'sorry',
        ],
        [
            'anyone', 'information', 'about', 'upcoming', 'computers', 'Cyclone', 'Tempest', 'need',
            'info', 'Anything', 'would', 'greatly', 'appreciated', 'Thanks', 'Shawn',
        ],
        [
            'Distribution', 'article', 'seanmcdacdalca', 'writes', 'article', 'Andre', 'Molyneux', 'writes',
            'David', 'Joshua', 'Mirsky', 'writes', 'LCIII', 'recently', 'heard', 'interesting',
            'heard', 'that', 'LCIII', 'built', 'slot', 'PowerPC', 'chip', 'this',
            'heard', 'that', 'slot', 'same', 'slot', 'that', 'true', 'Thanks',
            'Mirsky', 'Well', 'also', 'have', 'Popping', 'revealed', 'socket', 'additional',
            'SIMM', 'socket', '72pin', 'SIMM', 'socket', 'flatpack', 'slot', 'identical',
            'with', 'additional', 'connetions', 'side', 'full', '32bit', 'data', 'path',
            'LCLC', 'lacked', 'Thats', 'guess', 'board', 'with', 'PowerPC', 'chip',
            'made', 'that', 'would', 'thats', 'only', 'place', 'will', 'possible',
            'NuBus', 'PowerPC', 'upgrade', 'will', 'require', 'logic', 'board', 'swap',
            'interesting', 'Apple', 'come', 'with', 'NuBus', 'PowerPC', 'that', 'allowed',
            '680x0', 'like', 'RocketShare', 'guess', 'thats', 'getting', 'fantastic', 'wondering',
            'MacWeek', 'reported', 'that', 'developers', 'were', 'seeded', 'with', 'PowerPCs',
            'card', 'Also', 'word', 'machine', 'arrivals', 'estimated', 'speed', 'Last',
            'estimates', 'were', 'around', 'times', 'speed', 'Quadra', 'native', 'RISC',
            'heard', 'Apple', 'employee', 'mumble', 'something', 'about', 'arrival', 'PowerPC',
            'much', 'earlier', 'date', 'that', 'doubt', 'true', 'Finally', 'PowerPC',
            'minicourse', 'available', 'advertised', 'developers', 'university', 'calendar', 'like', 'know',
            'Sean', 'seanmcdacdalca', 'Radius', 'speculated', 'publicly', 'that', 'they', 'could',
            'PowerPCbased', 'Rocket', 'existing', 'Macs', 'would', 'have', 'plus', 'RocketShare',
            'NuBus', 'accelerators', 'true', 'boot', 'accelerator', 'NuBus', 'bottleneck', 'video',
            'Apple', 'seems', 'will', 'compete', 'with', 'third', 'parties', 'here',
            'perhaps', 'Macs', 'like', 'Cyclone', 'where', 'PowerPC', 'slot', 'might',
            'Look', 'Daystar', 'such', 'make', 'PowerPC', 'accelerators', 'potential', 'problem',
            'accelerator', 'though', 'that', 'will', 'need', 'companion', 'Apple', 'licensed',
            'Radius', 'with', 'Rocketshare', 'proprietary', 'code', 'Apple', 'between', 'lines',
            'know', 'that', 'PowerPC', 'Macs', 'will', 'have', 'simplified', 'logic',
            'magical', 'nature', 'RISC', 'that', 'these', 'boards', 'should', 'much',
            'build', 'than', 'those', 'existing', '68040', 'Macs', 'Perhaps', 'then',
            'groundbreaking', 'prices', 'Maclogic', 'board', 'upgrades', 'much', 'same', 'weve',
            'much', 'cheaper', 'CPUs', 'this', 'year', 'First', 'generation', 'PowerPCs',
            'will', 'also', 'hopefully', 'have', 'socketed', 'CPUs', 'that', 'theyll',
            'upgradeable', '98604s', 'year', 'later', 'This', 'should', 'possible', 'much',
            'that', '486s', 'pulled', 'clock', 'doublers', 'there', 'much', 'technical',
            'which', 'doubt', 'since', 'external', 'busses', 'same', 'sizewidth', 'this',
            'have', 'daughterboard', 'Powerbook', 'standard', 'facilitate', 'better', 'upgrades', 'This',
            'where', 'Apple', 'fallen', 'behing', 'Intelbased', 'world', 'Perhaps', 'catchup',
            'last', 'weeks', 'week', 'excellent', 'story', 'PowerPC', 'Pentium', 'MIPS',
            'Alpha', 'four', 'microprocessor', 'front', 'forseeable', 'future', 'Worth', 'reading',
            'Also', 'latest', 'cover', 'story', 'Pentium', 'Read', 'other', 'stories',
            'Intel', 'unstoppable', 'preeminent', 'right', 'Once', 'anyone', 'this', 'secure',
            'fall', 'Intels', 'market', 'position', 'will', 'never', 'again', 'dominant',
            'especially', 'gets', 'ahead', 'sell', '486s', 'this', 'week', 'appears',
            'competition', 'from', 'fronts', 'gearing', 'awesome', 'battle', 'Apple', 'users',
            'excited', 'that', 'PowerPC', 'while', 'guaranteed', 'dominance', 'guaranteed', 'winner',
            'several', 'Mark',
        ],
        [
            'borrowed', '199293', 'version', 'this', 'book', 'from', 'friendholy', 'moley',
            'wealth', 'contacts', 'Fivehundred', 'pages', 'information', 'about', 'electronic', 'artists',
            'around', 'globe', 'many', 'have', 'email', 'addresses', 'minute', 'database',
            'information', 'also', 'available', 'Minitel', 'books', 'based', 'Franceare', 'there',
            'book', 'printed', 'French', 'English', 'have', 'your', 'organization', 'listed',
            'just', 'send', 'your', 'information', 'Annick', 'Bureaud', 'IDEA', 'Falguiere',
            'Paris', 'France', 'free', 'listed', 'sure', 'widely', 'distributed', 'book',
            'costs', 'affiliated', 'with', 'them', 'just', 'impressed', 'their', 'collection',
            'artists', 'highly', 'encourage', 'involved', 'electronic', 'media', 'video', 'music',
            'animation', 'send', 'your', 'entry', 'encourage', 'them', 'make', 'their',
            'available', 'Internet', 'Stastny', 'OTIS', 'Project', 'PROCESS', 'SOUND', 'News',
            '241113', 'sunsiteuncedu', 'Omaha', '681241113', '1412144135', 'projectsotis', 'EMail',
        ],
        [
            'Hello', 'everybody', 'using', 'PIXARS', 'RenderMan', 'scene', 'description', 'language',
            'worlds', 'please', 'help', 'using', 'RenderMan', 'library', 'NeXT', 'there',
            'about', 'NeXTSTEP', 'version', 'RenderMan', 'available', 'create', 'very', 'complicated',
            'render', 'them', 'using', 'surface', 'shaders', 'bring', 'them', 'life',
            'shadows', 'reflections', 'understand', 'have', 'define', 'environmental', 'shadows', 'maps',
            'reflections', 'shadows', 'know', 'them', 'advises', 'simple', 'examples', 'will',
            'Thanks', 'advance', 'Alex', 'Kolesov', 'Moscow', 'Russia', 'Talus', 'Imaging',
            'Corporation', 'email', 'alextalusmsksu', 'NeXT', 'mail', 'accepted',
        ],
        [
            'article', 'David', 'Joshua', 'Mirsky', 'writes', 'LCIII', 'recently', 'heard',
            'rumor', 'heard', 'that', 'LCIII', 'built', 'slot', 'PowerPC', 'chip',
            'true', 'heard', 'that', 'slot', 'same', 'slot', 'that', 'true',
            'David', 'Mirsky', 'Well', 'also', 'have', 'Popping', 'revealed', 'socket',
            'VRAM', 'SIMM', 'socket', '72pin', 'SIMM', 'socket', 'flatpack', 'slot',
            'LCLC', 'with', 'additional', 'connetions', 'side', 'full', '32bit', 'data',
            'that', 'LCLC', 'lacked', 'Thats', 'guess', 'board', 'with', 'PowerPC',
            'could', 'made', 'that', 'would', 'thats', 'only', 'place', 'Andre',
            'KA7WVV', 'Insert', 'your', 'favorite', 'disclaimer', 'here', 'PYRAMID', 'TECHNOLOGY',
            'Internet', '3860', 'First', 'Street', 'Jose', 'Packet', '4288229',
        ],
        [
            'Rene', 'Walter', 'writes', 'very', 'kind', 'soul', 'mailed', 'this',
            'bugs', 'CView', 'Since', 'isnt', 'position', 'post', 'this', 'himself',
            'post', 'leave', 'name', 'here', 'comes', 'CView', 'quite', 'number',
            'mention', 'perhaps', 'most', 'stupid', 'question', 'what', 'will', 'CView',
            'still', 'need', 'viewer', 'Linux', 'Without', 'XWindows', 'Thanks',
        ],
        [
            'article', 'Wtte', 'writes', 'have', 'friends', 'optical', '128M', 'optical',
            'lose', 'friends', 'they', 'smell', 'stop', 'worrying', 'about', 'cartridge',
            'Bernoulli', 'crashes', 'SyQuest', 'serious', 'note', 'have', 'heard', 'tales',
            'SyQuest', 'failures', 'curious', 'about', 'Jons', 'comments', 'cartridge', 'wear',
            'someone', 'elaborate', 'there', 'general', 'consensus', 'that', '128M', 'opticals',
            'reliable', 'mostly', 'concerned', 'about', 'media', 'failures', 'opposed', 'drive',
            'failures', 'Julian', 'Vrieslander', 'Neurobiology', 'Behavior', 'Mudd', 'Hall', 'Cornell',
            'Ithaca', '14853', 'INTERNET', 'BITNET', 'eacjcrnlthry', 'UUCP',
        ],
        [
            'article', 'mbcpoCWRUEdu', 'Michael', 'Comet', 'wrote', 'previous', 'article', 'Tony',
            'says', 'There', 'product', 'IBMers', 'there', 'called', 'IMAGINE', 'just',
            'shipping', 'yesterday', 'personally', 'attest', 'that', 'will', 'blow', 'doors',
            'made', 'IMPUlSE', 'WellI', 'dont', 'know', 'about', 'competing', 'with',
            'pretty', 'powerful', 'allright', 'issue', '_SPEED_', 'fast', 'Imagine', 'easy',
            'just', 'render', 'fast', 'dont', 'want', 'things', 'like', 'fine',
            'animated', 'reflection', 'maps', 'animated', 'bump', 'maps', 'animated', 'anything',
            'with', 'IPAS', 'routines', 'that', 'ever', 'seen', 'them', 'explosions',
            'morphing', 'fire', 'rain', 'lens', 'flares', 'knocking', 'imagine', 'just',
            'know', 'compares', 'with',
        ],
        [
            'kavitskyhsicom', 'Does', 'anyone', 'know', 'high', 'order', 'being', 'filtered',
            'make', 'sure', 'that', 'entire', '8bits', 'make', 'through', 'final',
            'help', 'greatly', 'appreciated', 'need', 'these', 'resources', 'true', 'true',
            'need', 'stty', 'istrip', 'Good', 'luck', 'Victor', 'Victor', 'Gattegno',
            'HewlettPackard', 'France', 'Hpdesk', 'HP8101RC', 'Avenue', 'Canada', 'Phone', '33169826060',
            'Ulis', 'Cedex', 'Telnet', '7701141',
        ],
        [
            'article', 'Nathan', 'Moore', 'writes', 'Nilay', 'Patel', 'writes', 'looking',
            'removable', 'tapes', '2020', 'drive', 'Dont', 'laugh', 'serious', 'have',
            'lying', 'around', 'that', 'would', 'like', 'please', 'mail', 'Nilay',
            'mean', 'disks', 'dont', 'tapes', 'forgot', 'whether', 'were', 'looking',
            'WellI', 'need', 'disks', 'right', 'disks', 'better', 'word', 'they',
            'them', 'disks', 'kind', 'funny', 'appropriate', 'word', 'disks', 'Nilay',

        ],
        [
            'currently', 'using', 'POVRay', 'wondering', 'anyone', 'netland', 'knows', 'public',
            'antialiasing', 'utilities', 'that', 'skip', 'this', 'step', 'very', 'slow',
            'machine', 'suggestions', 'opinions', 'about', 'posttrace', 'antialiasing', 'would', 'greatly',
            'Helmut', 'Dotzlaw', 'Dept', 'Biochemistry', 'Molecular', 'Biology', 'University', 'Manitoba',
            'Canada',
        ],
        [
            'article', 'Latonia', 'writes', 'cant', 'imagine', 'someone', 'would', 'leave',
            'computer', 'time', 'start', 'with', 'like', 'leaving', 'your', 'lights',
            'everything', 'house', 'time', 'meNuts', 'Computers', 'special', 'case', 'pretty',
            'idea', 'leave', 'them', 'everytime', 'turn', 'computer', 'youre', 'putting',
            'electricity', 'through', 'delicate', 'components', 'Imagine', 'youre', 'turning', 'your',
            'more', 'times', 'Youre', 'increasing', 'chances', 'damaging', 'chips', 'memory',
            'your', 'computer', 'save', 'cents', 'here', 'there', 'electricity', 'bills',
            'look', 'like', 'much', 'when', 'come', 'time', 'your', 'computer',

        ],
        [
            'vm1mcgillca', 'Organization', 'McGill', 'University', 'Hello', 'having', 'small', 'problem',
            'sound', 'blaster', 'game', 'there', 'utility', 'there', 'that', 'would',
            'what', 'DMAs', 'system', 'using', 'Thanks', 'Mark', 'Brown',
        ],
        [
            'wonder', 'possible', 'parent', 'window', 'paint', 'over', 'area', 'childs',
            'could', 'possible', 'implement', 'rubberband', 'across', 'multiple', 'xwindows', 'select',
            'that', 'displayed', 'each', 'window', 'Hauke',
        ],
        [
            'Distribution', 'world', 'Keywords', 'tvtwm', 'article', 'Mouse', 'writes', 'article',
            'David', 'Simon', 'writes', 'some', 'please', 'explain', 'following', 'piece',
            'causes', 'tvtwm', 'dump', 'core', 'particular', 'interested', 'knowing', 'whether',
            'behavior', 'caused', 'reasoning', 'anything', 'client', 'does', 'causes', 'dump',
            'Window', 'managers', 'should', 'never', 'ever', 'crash', 'Would', 'only',
            'true', 'only', 'would', 'crash', 'once', 'then', 'could', 'that',
            'unable', 'crash', 'either', 'tvtwm', 'which', 'would', 'remarkable', 'feat',
            'desirable', 'boot', 'mean', 'this', 'only', 'been', 'reported', 'zillion',
            'servers', 'other', 'hand', 'want', 'crash', 'OpenWindows', 'xnews', 'server',
            'Just', 'xbiff', 'Blammo', 'Greg', 'Earle', 'Phone', '3538695', '3531877',
            'UUCP',
        ],
        [
            'there', 'version', 'that', 'been', 'ported', 'Solaris', 'including', 'ANSI',
            'problems', 'trying', 'compile', 'under', 'Solaris', 'functions', 'have', 'prototypes',
            'from', 'User', 'Groups', '1992', 'Please', 'email', 'answers', 'this',
            'Tulinsky', 'Capital', 'Management', 'Sciences', 'West', 'Angeles', '9715', 'MANUALLY',
            'answers',
        ],
        [
            'Cutsie', 'little', 'Macintrashlike', 'icons', 'that', 'instant', 'recipe', 'mousitis',
            'System', 'undoubtedly', 'worst', 'have', 'used', 'that', 'RISCOS', 'MSWombles',
            'because', 'does', 'provide', 'enough', 'keyboard', 'shortcuts', 'Windows', 'must',
            'quite', 'like', 'cover', 'your', 'ears', 'because', 'actually', 'without',
            'ever', 'touch', 'mouse', 'stuff', 'delete', 'user', 'rather', 'than',
            'things', '_easier_', 'there', 'should', 'always', 'option', 'your', 'want',
            'like', 'UNIXX', 'combination', 'much', 'customizable', 'Hear', 'Hear', 'agree',
            'thing', 'cant', 'stand', 'about', 'interface', 'shear', 'determination', 'FORCE',
            'your', 'mouse', 'breaksyour', 'whole', 'system', 'down', 'like', 'mouseit',
            'some', 'occassions', 'such', 'past', 'moving', 'icons', 'around', 'most',
            'keyboard', 'keys', '1020', 'times', 'faster', 'than', 'using', 'mouse',
            'plus', 'able', 'something', 'simple', 'inexperienced', 'user', 'long', 'before',
            'experienced', 'month', 'Speaking', 'moment', 'dont', 'think', 'much', 'that',
            'programmers', 'provide', 'only', 'menumouse', 'interface', 'also', 'look', 'forward',
            'would', 'like', 'move', 'keys', 'command', 'line', 'interfaces', 'which',
            'allows', 'more', 'less', 'time', 'experienced', 'above', 'equally', 'applies',
            'systems', 'UNIX', 'especially', 'since', 'Unix', 'least', 'more', 'powerful',
            'Seth', 'Buffington', '8175652642', 'sethcsuntedu', 'sethgabuntedu', 'Unix', 'Operator',
        ],
        [
            'many', 'recent', 'advertisements', 'have', 'seen', 'both', '486DX50', '486DX',
            'systems', 'Does', 'first', 'really', 'exists', 'does', 'imply', 'that',
            'motherboard', 'with', 'works', 'that', 'speed', 'opposite', 'AYlatter', 'where',
            'internals', 'working', '50MHz', 'Many', 'thanx', 'advance', 'AYAndrew', 'Andrew',
            'version', '50MHz', 'considering', 'buying', 'other', 'definitely', 'with', 'nice',
            'external', 'cache', 'performance', 'greater', 'only', 'internal', 'cache', 'work',
            '50MHz', 'while', 'potentially', 'much', 'larger', 'cache', 'work', '50MHz',
            'Neither', 'systems', 'could', 'actually', 'program', 'main', 'memory', 'since',
            'still', 'slow', 'that', 'high', 'speed', '60ns', '1666MHz', '50MHz',
            '20b0', 'Unregistered', 'Evaluation', 'Copy', 'KMail', '295d', 'WNET', '4173',
            '9000', 'QWKtoUsenet', 'gateway', 'Four', '14400', 'v32bis', 'dialins', 'FREE',
            'mail', 'newsgroups', 'PCBoard', '145aM', 'uuPCB', 'Kmail', 'Call', '4173',
            'Member', 'ASAD', '1500MB', 'disk', 'Serving', 'Arbor', 'since', '1988',
        ],
        [
            'Hello', 'Netlanders', 'novice', 'user', 'with', 'question', 'Xgod', 'computer',
            'with', 'problem', 'follows', 'running', 'Esix', 'Wangtek', 'ATstyle', 'interface',
            'drive', 'have', 'loaded', 'Basic', 'which', 'includes', 'inet', 'utilities',
            'ftped', 'XFree86', 'X11R5', 'binaries', 'installed', 'properly', 'execute', 'startx',
            'with', 'problems', 'However', 'access', 'tape', 'drive', 'while', 'machine',
            'instantly', 'access', 'tape', 'tape', 'drive', 'works', 'fine', 'Soon',
            'again', 'screen', 'changes', 'modes', 'grey', 'background', 'pattern', 'does',
            'xterm', 'forked', 'have', 'login', 'from', 'another', 'terminal', 'execute',
            'reset', 'system', 'contacted', 'Esix', 'about', 'this', 'problem', 'They',
            'THEIR', 'Xwindow', 'X11R4', 'server', 'which', 'have', 'works', 'with',
            'tape', 'drive', 'They', 'also', 'claim', 'only', 'need', 'network',
            'utilities', 'dont', 'need', 'inet', 'tcpip', 'experience', 'been', 'that',
            'BOTH', 'XFree86', 'work', 'concerned', 'about', 'having', 'load', 'both',
            'packages', 'work', 'unless', 'inet', 'package', 'causing', 'problem', 'would',
            'both', 'tape', 'drive', 'coexist', 'same', 'system', 'shed', 'light',
            'would', 'appreciated', 'colleague', 'implied', 'this', 'might', 'hardware', 'conflict',
            'true', 'what', 'direction', 'should', 'look', 'resolve', 'conflict', 'Thanks',
            'Cobler', 'kscihlpvattcom', 'Bell', 'Laboratories', 'Shuman', 'Blvd', 'Naperville', '60566',
        ],
        [
            'article', 'writes', 'However', 'that', 'almost', 'overkill', 'Something', 'more',
            'this', 'would', 'probably', 'make', 'EVERYONE', 'happier', 'Thats', 'closer',
            'apps', 'software', 'hardware', 'would', 'better', 'Would', 'that', 'engulf',
            'that', 'programmer', 'dont', 'know', 'traffic', 'really', 'heavy', 'enough',
            'newsgroup', 'split', 'Look', 'busy', 'true', 'that', 'traffic', 'here',
            'FAQs', 'discussing', 'things', 'that', 'would', 'probably', 'better', 'diverted',
            'groups', 'dont', 'know', 'whether', 'split', 'would', 'help', 'hurt',
            'cause', 'Maybe', 'need', 'those', 'people', 'cant', 'bothered', 'read',
            'books', 'there', 'Right', 'Rogers',
        ],
        [
            'Dear', 'netters', 'have', 'noticed', 'something', 'rather', 'weared', 'think',
            'creating', 'dialog', 'shell', 'widget', 'while', 'running', 'Vues', 'vuewm',
            'reason', 'every', 'time', 'create', 'dialog', 'shell', 'foreground', 'backgroun',
            'different', 'compared', 'toplevel', 'shell', 'doing', 'anything', 'Does', 'body',
            'anything', 'about', 'this', 'problem', 'without', 'hardcodin', 'colors', 'Please',
            'Thanks', 'Kamlesh',
        ],
        [
            'Distribution', 'world', 'article', 'mahanTGVCOM', 'Patrick', 'Mahan', 'writes', 'gotten',
            'posts', 'this', 'group', 'last', 'couple', 'days', 'recently', 'added',
            'list', 'just', 'this', 'group', 'near', 'death', 'Seen', 'from',
            'list', 'side', 'getting', 'about', 'right', 'amount', 'traffic', 'seen',
            'point', 'view', 'much', 'articles', 'keep', 'with', 'them', 'lucky',
            'through', 'subjects', 'from', 'time', 'time', 'DiplInform', 'Rainer', 'Klute',
            'richtig', 'beraten', 'Univ', 'Dortmund', 'Postfach', '500500', '7554663', 'DW4600',
            '7552386', 'address', 'after', 'June', '30th', 'Univ', 'Dortmund', 'D44221',

        ],
        [
            'Distribution', 'world', 'dantenmsuedu', 'Wayne', 'Smith', 'writes', 'This', 'doesnt',
            'original', 'question', 'multiuser', 'mention', 'made', 'ether', 'card', 'either',
            'diskdata', 'point', 'view', 'does', 'SCSI', 'have', 'advantage', 'when',
            'multi', 'tasking', 'Data', 'data', 'could', 'anywhere', 'drive', 'SCSI',
            'faster', 'drive', 'into', 'computer', 'faster', 'Does', 'have', 'better',
            'system', 'thought', 'SCSI', 'good', 'managing', 'data', 'when', 'multiple',
            'attached', 'only', 'talking', 'about', 'single', 'drive', 'explain', 'SCSI',
            'faster', 'managing', 'data', 'from', 'hard', 'drive', 'making', 'same',
            'confusing', 'DRIVE', 'interface', 'DATA', 'THROUGHPUT', 'interface', 'Again', 'from',
            'sheet', 'available', '364406', 'infomacreport', 'Expansion', 'Both', 'SCSI', 'only',
            'device', 'expansion', 'interface', 'common', 'both', 'Allows', 'device', 'hard',
            'printer', 'scanner', 'Nubus', 'card', 'expansion', 'Plus', 'only', 'some',
            'CDROM', 'Apple', 'developed', 'some', 'specifications', 'SCSI', 'controlers', 'while',
            'controller', 'specifications', 'which', 'results', 'added', 'machines', 'Main', 'problem',
            'external', 'devices', 'which', 'internal', 'terminated', 'which', 'causes', 'problems',
            'then', 'devises', 'SCSI', 'port', 'SCSI', 'chain', 'supposed', 'terminated',
            'begining', 'other', 'causes', 'problems', 'either', 'SCSI1', 'devices', 'SCSI',
            '8bit', 'asynchronous', '15MBs', 'synchronous', '5MBs', 'transfer', 'base', '16bit',
            'requires', 'SCSI2', 'controler', 'chip', 'provide', 'only', 'fast', 'SCSI2',
            'SCSI2', 'which', 'both', '16bit', 'interfaces', 'SCSI2', 'SCSI2', 'devices',
            'controller', 'SCSI2', 'mode', 'SCSI2', 'fully', 'SCSI1', 'complient', 'tends',
            'very', 'fast', 'SCSI1', 'since', 'needs', 'different', 'controller', 'interface',
            'hardware', 'which', 'tends', 'very', 'expendsive', 'software', 'Transfer', 'speeds',
            'with', '10MBs', 'burst', '8bit', '812MBs', 'with', '20MBs', 'burst',
            '1520MBs', 'with', '40MBs', 'burst', '32bitwide', 'fast', 'SCSI2', 'SCSI1',
            'limited', 'devices', 'reduced', '8bit', '16bit', 'fast', 'only', 'throughput',
            'between', 'SCSI1', 'wide', 'SCSI2', 'ports', 'Interfaces', 'limited', 'hard',
            'design', 'lack', 'development', 'Integrated', 'Device', 'Electronics', 'currently', 'most',
            'standard', 'mainly', 'used', 'medium', 'sized', 'drives', 'have', 'more',
            'hard', 'drive', 'Asynchronous', 'Transfer', '5MBs', 'LOWEST', 'setting', 'SCSI2',
            'Asynchronous', 'SCSI1', 'mode', 'AVERAGES', 'through', 'MAXIMUM', 'asynchronous', 'mode',
            'SCSI2', 'mode', 'blows', 'poor', 'window', 'down', 'street', 'into',
            'problem', 'becomes', 'drive', 'mechanisim', 'keep', 'with', 'those', 'through',
            'THAT', 'where', 'bottleneck', 'cost', 'SCSI2', 'comes', 'from', 'interface',
            'more', 'more', 'from', 'drive', 'mechanisims', 'SCSI2', 'through', 'cost',
            'interface', 'self', 'fulliling', 'prophisy', 'people', 'SCSI', 'because', 'expencive',
            'turn', 'convices', 'makes', 'that', 'mass', 'producing', 'SCSI', 'which',
            'reduce', 'cost', 'unwarented', 'SCSI', 'expencive', 'That', 'effect', 'Rule',
            'more', 'items', 'sold', 'less', 'EACH', 'item', 'bare', 'brunt',
            'manufacture', 'less', 'each', 'item', 'cost', 'SCSI2', 'allows', 'drive',
            'through', 'limited', 'DRIVE', 'while', 'itself', 'limits', 'through',
        ],
        [
            'everybody', 'anyone', 'name', 'anonymous', 'ftpsite', 'where', 'find', 'sources',
            'package', 'portable', 'bitgraypixel', 'would', 'like', 'compile', 'Sparcstation', 'Thanks',
        ],
        [
            'article', 'Oliver', 'Duesel', 'writes', 'there', 'Yuri', 'Yulaev', 'writes',
            '1s1p1g', 'card', '38640', 'When', 'plug', 'wang', 'modem', 'com4it',
            'change', 'com1', 'doesnt', 'Program', 'chkport', 'gives', 'diagnostics', 'like',
            'conflict', 'com1', 'with', 'mouse', 'driver', 'memory', 'Since', 'your',
            'only', 'serial', 'port', 'this', 'should', 'default', 'COM1', 'Under',
            'cant', 'share', 'IRQs', 'youll', 'have', 'either', 'your', 'modem',
            'mouse', 'COM2', 'using', 'different', 'adresses', 'IRQs', 'When', 'devices',
            'same', 'like', 'COM1', 'COM3', 'latter', 'will', 'always', 'have',
            'mouse', 'COM1', 'start', 'using', 'your', 'modem', 'COM3', 'your',
            'should', 'work', 'your', 'mouse', 'will', 'stop', 'doing', 'until',
            'should', 'problem', 'setting', 'your', 'modem', 'COM2', 'didnt', 'write',
            'about', 'other', 'peripherals', 'hope', 'helped', 'Byte', 'kind', 'stuff',
            'serial', 'ports', 'and3', 'share', 'same', 'IRQs', 'mean', 'cant',
            'mouse', 'into', 'Com1', 'modem', 'into', 'com3', 'expect', 'both',
            'Answer', 'should', 'change', 'IRQs', 'ports', 'different', 'does', 'really',
            'which', 'ports', 'Phil', 'Phil', 'Hunt', 'Wherever', 'there', 'Howtek',
            'uucp',
        ],
        [
            'those', 'interested', 'Traced', 'pictures', 'there', 'nice', 'example', 'file',
            'Poolballgif', 'shows', 'pooltable', 'with', 'poolballs', 'Resolution', '1024x768', 'colours',
            'version', 'also', 'available', '24Mb', 'post', 'picture', 'created', 'with',
            'Enjoy', '_Gerco_', 'Gerco', 'Schot', 'cgschotcsruunl',
        ],
    ],
}

print(data.keys())
