#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
#


import json
import cPickle as pickle
import numpy as np
import h5py
import random
import pandas as pd
from nltk.tokenize import TweetTokenizer
word_tokenize = TweetTokenizer().tokenize
import re


# IMPORTANT: Make sure the parameters below match the specification of the generated
# summaries (i.e. the params['summaries_filename'] variable) in terms of the state and
# and the dataset (i.e. params['dataset_location']) that will be loaded.
params = {
    'state': 'test',
    # 'state': 'validate',
    'dataset_location': '../Datasets/ar/with_property_placeholders/',
    # 'summaries_filename': './checkpoints/eo/with_property_placeholders/surf_form_tuples.model.t7.batch_size_85.beam_size_20.summaries_Testing.h5'
    # 'summaries_filename': './checkpoints/eo/without_property_placeholders/surf_form_tuples.model.t7.batch_size_85.beam_size_20.summaries_Validation.h5'
    'summaries_filename': './checkpoints/ar/with_property_placeholders/surf_form_tuples.model.t7.batch_size_85.beam_size_20.summaries_Testing.h5'
    # 'summaries_filename': './checkpoints/ar/without_property_placeholders/surf_form_tuples.model.t7.batch_size_85.beam_size_20.summaries_Testing.h5'
}
labels_file_location = '../Datasets/ar/Labels/labels_dict.p'
# We are only be displaying the most probable summary.
beamidx = 0

# The location that the output .csv will be stored.
summaries_dump_location = params['summaries_filename'].replace('h5', 'p')
# IMPORTANT: Leave the batch size unchanged
# It's the one with which we trained the models, and it should be the same
# with the one of the loaded pre-trained model that was used to generate the summaries
# (i.e. with beam-sample.lua). Change only if you train your own models using a
# different batch size.
batch_size = int(re.findall(r'(?<=batch_size_)(.*)(?=.beam_size)', params['summaries_filename'])[0])
beam_size = int(re.findall(r'(?<=beam_size_)(.*)(?=.summaries)', params['summaries_filename'])[0])


print('Parameters')
for key in params:
    print('%s: %s' % (key, params[key]))



# Loading relevant dataset files.
summaries = h5py.File(params['summaries_filename'], 'r')


with open(params['dataset_location'] + 'summaries_dictionary.json', 'r') as f:
    summaries_dictionary = json.load(f, 'utf-8')
    id2word = summaries_dictionary['id2word']
    id2word = {int(key): id2word[key] for key in id2word}
    word2id = summaries_dictionary['word2id']   
    
with open(params['dataset_location'] + 'triples_dictionary.json', 'r') as f:
    triples_dictionary = json.load(f, 'utf-8')
    max_num_triples = triples_dictionary['max_num_triples']
    id2item = triples_dictionary['id2item']
    id2item = {int(key): id2item[key] for key in id2item}
    item2id = triples_dictionary['item2id']
    
    
# Loading supporting inverse dictionaries for surface forms and instance types.
with open(params['dataset_location'] + 'inv_surf_forms_dictionary.json', 'r') as f:
    inv_surf_forms_tokens = json.load(f, encoding='utf-8')
with open(params['dataset_location'] + 'surf_forms_counts.p', 'rb') as f:
    surf_forms_counts = pickle.load(f)
        
with open(params['dataset_location'] + 'inv_instance_types_with_predicates.json', 'r') as f:
    inv_instancetypes_with_pred_dict = json.load(f, encoding='utf-8')
with open(params['dataset_location'] + 'splitDataset_with_targets.p', 'rb') as f:
    splitDataset = pickle.load(f)

# Loading supporting labels_en dataset.
with open(labels_file_location, 'rb') as f:
    labels = pickle.load(f)

print('All relevant dataset files from: %s have been successfully loaded.' % params['dataset_location'])


# Example of the structure of the supporting dictionaries:
# surf_form_counts[u'http://www.wikidata.org/entity/Q46611']: {u'Apollo-Programo': 10, u'Projekto Apollo': 6, u'projekto Apollo': 2}
# inv_surf_forms_tokens[u'#surFormToken71849']: [u'http://www.wikidata.org/entity/Q832222', u'Caprivi-streko']
# inv_instancetypes_with_pred_dict[u'#instanceTypeWithPredicate11']: u'http://www.wikidata.org/prop/direct/P138'


most_frequent_surf_form = {}
for entity in surf_forms_counts:
    most_frequent_surf_form[entity] = sorted(surf_forms_counts[entity], key=lambda k: surf_forms_counts[entity][k], reverse=True)[0]



def tokenizeNumbers(inp_string):
    tokens = word_tokenize(inp_string)
    for j in range(0, len(tokens)):
        try:
            tempNumber = float(tokens[j].replace(',', ''))
            if tempNumber // 1000 >= 1 and tempNumber // 1000 < 3:
                tokens[j] = '<year> '
            else:
                tokens[j] = '0 '
        except ValueError:
            pass
    # return detokenize(tokens, return_str=True) # detokenize has an issue with the non-latin characters.
    return ' '.join(tokens)


def match_predicate_to_entity(token, triples, expressed_triples):
    matched_entities = []
    
    for tr in range(0, len(triples)):
        if tr not in expressed_triples:
            tempPredicate = triples[tr].split()[1]
            if tempPredicate == token:
                tempEntity = triples[tr].split()[-1]
                if tempEntity == "<item>":
                    tempEntity == triples[tr].split()[0]
                if tempEntity not in matched_entities:
                    matched_entities.append(tempEntity.decode('utf-8'))

    if len(matched_entities) == 0:
        token = '<resource>'
    else:
        
        random_selection = random.choice(matched_entities)
        while random_selection not in labels and len(matched_entities) > 1:
            matched_entities.remove(random_selection)
            random_selection = random.choice(matched_entities)
        if random_selection in labels:
            if 'Datasets/ar/' in labels_file_location:
                token = labels[random_selection].decode('unicode-escape')
            else:
                token = labels[random_selection]
            expressed_triples.append(random_selection)
        else:
            token = '<resource>'
    return token


def token_to_word(token, main_entity, triples, expressed_triples):
    global summaries_type
    
    if 'without_property_placeholders' in params['summaries_filename']:
        assert ('#instanceTypeWithPredicate' not in token)
    
    main_entity = main_entity
    if "#surFormToken" in token:
        word = inv_surf_forms_tokens[token[1:]][1] if "##surFormToken" in token else inv_surf_forms_tokens[token][1]
    elif "#instanceTypeWithPredicate" in token:
        word = match_predicate_to_entity(inv_instancetypes_with_pred_dict[token], triples, expressed_triples)
    elif "#instanceType" in token:
        word = inv_instancetypes_dict[token]
    elif token == "<item>":
        # The returned variable word is of type: unicode.
        word = tokenizeNumbers(most_frequent_surf_form[main_entity])
        
    else:
        word = token
    return word



output = {'Main-Item': [],
          'index': [],
          'number_original_triples': [],
          'original_triples': [],
          'number_input_triples': [],
          'final_triples_with_types_reduced': [], 
          'final_triples_with_types': [], 
          'Target': [], 
          'Generated-Summary': []}

for batchidx in range(0, len(summaries['triples'])):
    print('Post-processing summaries from %d. Batch...' % (batchidx + 1))
    for instance in range(0, batch_size):
        # Pay attention to the Python division at the np.round() function -- can seriously mess things up!
        # More info at: https://stackoverflow.com/questions/28617841/rounding-to-nearest-int-with-numpy-rint-not-consistent-for-5
        # We are using the built-in version of round which seems to be doing the trick for now.
        splitDatasetIndex = int(round(instance * len(splitDataset[params['state']]['item']) / float(batch_size)) + batchidx)
        mainItem = splitDataset[params['state']]['item'][splitDatasetIndex].decode('utf-8')
        
        final_triples_with_types = []
        for tr in range(0, len(splitDataset[params['state']]['final_triples_with_types'][splitDatasetIndex])):
            
            tempTriple = splitDataset[params['state']]['final_triples_with_types'][splitDatasetIndex][tr] 
            if type(tempTriple) is not unicode:
                tempTriple = tempTriple.decode('utf-8')
            final_triples_with_types.append(tempTriple.replace('<item>', mainItem))
        
        final_triples_with_types_reduced = []
        for tr in range(0, len(splitDataset[params['state']]['final_triples_with_types_reduced'][splitDatasetIndex])):
            # eq_used_for_training_triple: the triple as it was used by the neural network
            # during training, validation and testing.
            eq_used_for_training_triple = ' '.join([id2item[summaries['triples'][batchidx][tr][instance][j]] for j in range(0, 3)])
            assert(splitDataset[params['state']]['final_triples_with_types_reduced'][splitDatasetIndex][tr] == eq_used_for_training_triple)
            if eq_used_for_training_triple is not unicode:
                eq_used_for_training_triple = eq_used_for_training_triple.decode('utf-8')
            final_triples_with_types_reduced.append(eq_used_for_training_triple.replace('<item>', mainItem))
        
        

        original_triples = []
        for tr in range(0, len(splitDataset[params['state']]['triples'][splitDatasetIndex])):
            tempTriple = splitDataset[params['state']]['triples'][splitDatasetIndex][tr]
            if type(tempTriple) is not unicode:
                tempTriple = tempTriple.decode('utf-8')
            original_triples.append(tempTriple.replace('<item>', mainItem))
        
        
        assert(len(final_triples_with_types) >= len(final_triples_with_types_reduced))
        assert(len(final_triples_with_types) == len(original_triples))
        
        expressed_triples = []
        # We read from the tail of the argsort to find the elements
        # with the highest probability.
        selected_summary_index = np.argsort(summaries['probabilities'][:, batchidx * batch_size + instance])[::-1][beamidx]
        summary = ''
        i = 0
        while summaries['summaries'][selected_summary_index][batchidx * batch_size + instance][i] != word2id['<end>']:
            summary += ' ' + token_to_word(id2word[summaries['summaries'][selected_summary_index][batchidx * batch_size + instance][i]],
                                           mainItem,
                                           splitDataset[params['state']]['triples'][splitDatasetIndex],
                                           expressed_triples)
            if i == len(summaries['summaries'][selected_summary_index][batchidx * batch_size + instance]) - 1:
                break
            else:
                i += 1
        summary += ' ' + token_to_word(id2word[summaries['summaries'][selected_summary_index][batchidx * batch_size + instance][i]],
                                       mainItem,
                                       splitDataset[params['state']]['triples'][splitDatasetIndex],
                                       expressed_triples)
        
        # Appending everything to the dictionary of lists.        
        if id2item[0] not in summary[1:]:
            output['index'].append((batchidx, instance))
            output['number_original_triples'].append(len(original_triples))
            output['original_triples'].append(original_triples)
            output['number_input_triples'].append(len(final_triples_with_types_reduced))
            output['final_triples_with_types_reduced'].append(final_triples_with_types_reduced)
            output['final_triples_with_types'].append(final_triples_with_types)
            output['Main-Item'].append(mainItem)
            output['Target'].append(splitDataset[params['state']]['actual_target'][splitDatasetIndex])
            output['Generated-Summary'].append(summary[1:])


# Saving all the generated summaries along with their input triples in a pickle file.
with open(summaries_dump_location, 'wb') as f:
    pickle.dump(output, f)
print('The generated summaries have been successfully saved at: %s' % summaries_dump_location)

