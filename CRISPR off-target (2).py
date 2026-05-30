#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# In[2]:


df = pd.read_csv('DataCRISPR-Cas9.csv.csv')
df


# In[107]:


df.shape


# In[3]:


def report(df):
    col = []
    d_type = []
    uniques = []
    n_uniques = []
    missing_values = []
    missing_percentage = []
    
    for i in df.columns:
        col.append(i)
        d_type.append(df[i].dtypes)
        uniques.append(df[i].unique()[:10])
        n_uniques.append(df[i].nunique())
        missing_values.append(df[i].isna().sum())
        missing_percentage.append(round(df[i].isna().sum()/len(df), 2)*100)
    
    return pd.DataFrame({'Column': col, 'dtype': d_type, 'unique sample': uniques, 'n uniques': n_uniques, 'num of missing': missing_values, 'mean of missing': missing_percentage})

report(df)


# In[4]:


df.columns


# Based on the outputs from `df.info()` and the `report(df)` summary above. We found:
# 
# * There are unneeded columns, we can use it just for filtring.
# columns like: `PMID` and `Year`
# 
# * There are unuseful columns like `Remark1`, `Remark2`, `Remark3` and `Remark4`.
# 
# * There are columns with duplicated infornmation like `Target_region` and `assembly_target_region`
# 
# * There are some columns with >50% missing values. The general rule is if a column has >50% missing consider dropping it. Otherwise impute (numeric: median or KNN; categorical: mode or a new category `Unknown`).
# 
# * There are some columns are `object` but contain numeric-like values like in `Indel_treatment%`, `Indel_accu%` and `Indel_control%`.
# 
# * Unnormlized unique values in some columns like in `Bulge` column and `Gene_symbol`.
# 
# * High cardinality problem or many unique values in categorical columns.
# 
# * Low cardinality problem like in `Success` problem. all column values = Success 
# 
# * Potential class imbalance in the target variable. We will handle it using stratified sampling, class weights, or resampling techniques (SMOTE, ADASYN) during training.
# 

# ## Data Cleaning and Processing
# 

# #### Feature Engineering and Selection
# 
# **Feature Selection** We will excude the columns that
# * Have duplicated information.
# * Have high missing percentage
# * With high/low cardinality
# * Have unuseful info like `Year` column
# 
# **Feature Engineering**
# We will create new columns that more informative and impactful or change column's values themself.
# * generate two new columns from `On_target_site`.
# * generate `destance` column from `start` and `end` column
# * process `Cas9_type` and `delivery` columns.
# 

# Drop remarks columns, not just because they don't have useful data but also they have >90% of missing data.

# In[5]:


# drop unuseful columns
df.drop(columns = ['Remark1', 'Remark2', 'Remark3', 'Remark4'], inplace=True)


# Drop columns with duplicated information, for example:
# 
# * `Target_region`               -> `chr` + `start` + `end`
# * `assembly_target_region`      -> `Assembly` + `chr` + `start` + `end`
# * `for_merge`                   -> `chr` + `start` + `end` + `Targeted_source` + `Assembly`
# * `CasRNA_browser`              -> `Cas9_type` + `gRNA`    

# In[6]:


df.drop(columns = ['Target_region','Targeted_delivery', 'assembly_target_region', 'for_merge', 'CasRNA_browser', 'Technology_others', 'tech_browser', 'source_browser'], inplace=True)


# Drop columns with high/low cardinality
# 
# The `Gene_OR_Locus` column contains 40% null values, while the remaining entries consist of 23,287 unique values.
# 
# The `success` column with only one value which is `success` and the rest is `null`.

# In[7]:


df.drop(columns = ['Gene_OR_Locus', 'success'], inplace=True)


# Drop columns with unuseful info. Just usefult in filteration

# In[8]:


df.drop(columns = ['PMID', 'Year'], inplace=True)


# Drop columns with high percentage of nulls
# 
# Above 90% of these columns values are null wgich means they aren't useful at all and can't be imputed. we will drop them.

# In[9]:


df.drop(columns = ['Targeted_method', 'Targeted_source', 'Targeted_location', 'Gene_symbol', 'Epigenetic_markers_combind', 'Indel_control%', 'Indel_treatment%', 'Indel_accu%', 'Validation','Bulge2', 'Bulge', 'Targeted_Time', 'Time'], inplace=True)


# #### `On_target_site` column

# In[10]:


df['On_target_site'].value_counts()


# We can split this column into 2 columns `on_target_gene` and `on_target_site`

# In[11]:


def split_site(s):
    parts = s.split('_')
    gene_name = parts[0]
    if len(parts) > 1:
        # site11 --> 11
        index_str = parts[1].replace("site", "")
        index_num = int(index_str)
    else: 
        index_num = None

    return (gene_name, index_num)

print(split_site('PDCD1_site17'))


# In[12]:


df['on_target_gene'], df['on_target_site_index'] = zip(*df['On_target_site'].apply(split_site))


# In[13]:


df['on_target_gene'].isna().sum(), df['on_target_site_index'].isna().sum()


# Drop `On_target_site` column

# In[14]:


df.drop(columns = ['On_target_site'], inplace=True)


# In[15]:


df['Species'].value_counts()


# there are 2 invalid categorical values. let's drop them. 

# In[16]:


# drop rows with invalid species
valid_species = ['Mus musculus', 'Homo sapiens']
df = df[df['Species'].isin(valid_species)]


# #### `Cas9_type` column
# 
# We found that the `Cas9_type` column contained multiple related enzymes (e.g., SpCas9, SaCas9, NmeCas9, Cas12a, etc.), so we grouped them based on their origin.
# 
# This helps reduce redundancy and allows the model to generalize better across similar enzyme types.
# 
# 

# In[17]:


df['Cas9_type'].value_counts()


# In[18]:


df['Cas9_type'].unique()


# In[19]:


SpCas9 = ['SpCas9', 'eSpCas9', 'SpCas9-HF1', 'SpRY', 'SniperCas9', 'xCas9(3.7)', 'SpCas9(K855A)', 'VP12Cas9', 'Alt-R HiFi Cas9', 'SpRY HF1', 
    'SpRY-Cas9', 'eSpCas9(1.1)', 'HypaCas9', 'HscCas9-v1.1', 
    'HscCas9-v1.2', 'SpCas9n', 'HiFi Cas9', 'LZ3 Cas9', 
    'evoCas9', 'SpCas9-mSA', 'SpCas9-mSA*', 
    'SpyCas9-mSA*', '3xNLS SpyCas9']
SaCas9 = ['SaCas9', 'SpCas9-SaCas9']
NmeCas9 = ['NmeCas9', 'NmCas9', 'Nme2Cas9', 'eNme2-C', 'eNme2-C.NR', 'SpCas9-NmCas9']
CjCas9 = ['CjCas9']
Cas12a = ['AsCpf1', 'AsCpf1_S542R/K607R', 'AsCpf1_S542R/K548V/N552R', 'LbCas12a', 'LbCas12a-T7', '3xNLS enAspCas12a']
BaseEditor = ['BE3', 'BE4max', 'ABE7.10', 'ABE8e', 'ABE8e-SpyMac', 'v5 AAV CBE', 'Sniper ABE7.10', 'ABE8.8']
PrimeEditor = ['PE2', 'PE3', 'PE4', 'PEmax-nuclease', 'PE2-nuclease']
CasMini = ['CasMINI-ge4.1', 'Un1Cas12f1-ge4.1']


# In[20]:


cas_mapping = {}
for val in SpCas9: cas_mapping[val] = 'SpCas9'
for val in SaCas9: cas_mapping[val] = 'SaCas9'
for val in NmeCas9: cas_mapping[val] = 'NmeCas9'
for val in CjCas9: cas_mapping[val] = 'CjCas9'
for val in Cas12a: cas_mapping[val] = 'Cas12a'
for val in BaseEditor: cas_mapping[val] = 'BaseEditor'
for val in PrimeEditor: cas_mapping[val] = 'PrimeEditor'
for val in CasMini: cas_mapping[val] = 'CasMini'

df['Cas_type'] = df['Cas9_type'].map(cas_mapping)


# In[21]:


df['Cas_type'].value_counts()


# drop `Cas9_type` column

# In[22]:


df.drop(columns = ['Cas9_type'], inplace=True)


# #### `Delivary` column
# 
# **`Delivary`** column contain many variations and typos (e.g., *“Lipofectamine”*, *“Lipofection”*, *“sgRNP Electroporation”*, etc.).
# 
# To reduce noise and improve consistency, we grouped similar terms into unified categories such as **Lipofectamine**, **PEI**, **Electroporation**, **Nucleofection**, and **Viral delivery**.
# 

# In[23]:


df['Delivery'].unique()


# In[24]:


delivery_map = {
    'lipofectamine': 'Lipofectamine',
    'lipofection': 'Lipofectamine',

    'polyethylenimine': 'PEI',
    'polyetherimide': 'PEI',
    'pei': 'PEI',

    'electroporation': 'Electroporation',
    'sgrnp electroporation': 'Electroporation',

    'nucleofection': 'Nucleofection',
    'nucleofector': 'Nucleofection',
    'sgrnp nucleofection': 'Nucleofection',

    'as3m.c lentivirus': 'Viral (Lentiviral)',
    'lentivirus tranduction': 'Viral (Lentiviral)',
    'oncolytic adenovirus': 'Viral (Adenoviral)',
    'adenoviral infection': 'Viral (Adenoviral)',
    'adenovirus': 'Viral (Adenoviral)',
    'aav9': 'AAV9',
    'aav-9': 'AAV9',
    'aav9 injection': 'AAV9',

    'microinjection': 'Microinjection',
    'htvi': 'HTVI',
    'hydrodynamic injection': 'HTVI',
    'intratracheally': 'Intratracheally',
    'subretinal injection': 'Subretinal injection',

    'vesicas spinoculation': 'Spinoculation (VEsiCas)',
    'vesicas': 'Spinoculation (VEsiCas)',
    'incubation': 'Incubation',
    'sgrnp incubation': 'Incubation',

    'transfection': 'Transfection',
    'co-transfection': 'Transfection',
    'co transfection': 'Transfection',
    'sgrna transfection': 'Transfection',
    'sgrnp transfection': 'Transfection',
    'linear dna transfection': 'Transfection',

    np.nan: 'Unknown'
}


# In[25]:


df['Delivery'] = df['Delivery'].str.lower().map(delivery_map)


# In[26]:


df['Delivery'].value_counts()


# #### Sequence columns

# In[27]:


df[['Guide_sequence', 'Target_sequence', 'Protospacer_sequence', 'PAM', 'Mismach', 'Mismatch_Pattern', 'PAM_Pattern', 'Sequence_show', 'Mismach2']].sample(15)


# As shown `Mismach2` calculated based on `Sequence_show`  (number of unmatched char between `Guide_sequence` and `Target_sequence` that represented in `Sequence_show` in small letters)
# 
# `Mismatch_Pattern` has many nulls and we can't recognize unmatched char (not represent all unmatched in small letters) so we will ignore him 
# 
# We will impute nulls in `Mismach2` using values in `Mismach` if the value exists
# 
# We can't find a goal to use `PAM_Pattern` column

# In[28]:


df['Mismach2'].isna().sum()


# In[29]:


df['Mismach2'] = df.apply(lambda row: row['Mismach'] if pd.isna(row['Mismach2']) else row['Mismach2'], axis=1)


# Let's drop unused columns

# In[30]:


df.drop(columns = ['Mismach', 'PAM_Pattern', 'Sequence_show', 'Mismatch_Pattern'], inplace=True)


# In[31]:


df.rename(columns={'Mismach2': 'Mismatches'}, inplace=True)


# #### `start` and `end` columns

# In[32]:


df['distance'] = df['end'] - df['start'] + 1


# In[33]:


df['distance'].value_counts()


# In[34]:


df['distance'][df['distance']==35444523.0]=22.0


# no need for `start` and `end` columns

# In[35]:


df.drop(columns = ['start', 'end'], inplace=True)


# ### Deal with nulls

# In[36]:


#Print only nulls columns with their counts
Report = report(df)


# In[37]:


Report[Report['num of missing']>0]


# We will impute `on_target_site_index` column with `0` 

# In[38]:


df['on_target_site_index'][df['on_target_site_index'].isna()]=0


# We notice that if `Identity` is `ON`, `Score` will be high and vice versa.
# 
# So, we will impute score based on `Identity` column.

# In[39]:


df['Score'] = df.groupby('Identity')['Score'].transform(lambda x: x.fillna(x.median()))


# In[40]:


df['distance'] = df['distance'].fillna(df['distance'].median())
df[['Strand', 'chr']] = df[['Strand', 'chr']].fillna(df[['Strand', 'chr']].mode().iloc[0])


# In[41]:


report(df)


# ## Filtering
# 
# We will focus only on HBB gene so let's Filter data.

# In[42]:


# Top target genes
gene_counts = df['on_target_gene'].value_counts().head(20)

plt.figure(figsize=(14, 6))
plt.barh(gene_counts.index, gene_counts.values, color='#8e44ad')
plt.xlabel('Count', fontsize=12, fontweight='bold')
plt.ylabel('Target Gene', fontsize=12, fontweight='bold')
plt.title('Top 15 Target Genes', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# In[43]:


#filter data to target only hbb gene
df = df[df['on_target_gene'] == 'HBB']


# In[44]:


report(df)


# Remove column with low cardinality like : `Species` and `on_target_gene`, Both have 1 value.

# In[45]:


df.drop(columns = ['Species', 'on_target_gene'], inplace=True)


# ## Data Visulization

# #### Distribution of Target Variable (Identity)

# In[46]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
identity_counts = df['Identity'].value_counts()
axes[0].bar(identity_counts.index, identity_counts.values, color=['#2ecc71', '#e74c3c'])
axes[0].set_xlabel('Identity', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0].set_title('Distribution of ON vs OFF Target Sites', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Pie chart
colors = ['#2ecc71', '#e74c3c']
axes[1].pie(identity_counts.values, labels=identity_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Proportion of ON vs OFF Target Sites', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


# #### Score Distribution by Identity

# In[47]:


# Score distribution for ON vs OFF targets
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Box plot
sns.boxplot(data=df, x='Identity', y='Score', palette=['#2ecc71', '#e74c3c'], ax=axes[0])
axes[0].set_title('Score Distribution by Identity', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Identity', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')

# Violin plot
sns.violinplot(data=df, x='Identity', y='Score', palette=['#2ecc71', '#e74c3c'], ax=axes[1])
axes[1].set_title('Score Distribution (Violin Plot)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Identity', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Score', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


# #### Mismatches Analysis

# In[48]:


mismatch_counts = df['Mismatches'].value_counts().sort_index()
plt.bar(mismatch_counts.index, mismatch_counts.values, color='#3498db')
plt.title('Overall Mismatches Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Number of Mismatches', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# #### Cas_type Analysis

# In[49]:


cas_counts = df['Cas_type'].value_counts()
plt.bar(cas_counts.index, cas_counts.values, color='#9b59b6')
plt.xlabel('Count', fontsize=12, fontweight='bold')
plt.ylabel('Cas Type', fontsize=12, fontweight='bold')
plt.title('Distribution of Cas Types', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()


# #### Delivery Method Analysis

# In[50]:


delivery_counts = df['Delivery'].value_counts().head(10)

# Top 10 delivery methods
plt.barh(delivery_counts.index, delivery_counts.values, color='#e67e22')
plt.xlabel('Count', fontsize=12, fontweight='bold')
plt.ylabel('Delivery Method', fontsize=12, fontweight='bold')
plt.title('Top 10 Delivery Methods', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

plt.show()


# ### 6. Species Distribution

# #### Assembly distribution

# In[51]:


assembly_counts = df['Assembly'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
axes[0].bar(assembly_counts.index, assembly_counts.values, color='#34495e')
axes[0].set_xlabel('Assembly', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0].set_title('Distribution of Genome Assembly', fontsize=14, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# Assembly by Identity
assembly_identity = df.groupby(['Assembly', 'Identity']).size().unstack(fill_value=0)
assembly_identity.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
axes[1].set_title('Assembly Distribution by Identity', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Assembly', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[1].legend(title='Identity', fontsize=10)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# #### Chromosome Distribution

# In[52]:


chr_counts = df['chr'].value_counts().head(15)

plt.figure(figsize=(14, 6))
plt.bar(range(len(chr_counts)), chr_counts.values, color='#1abc9c')
plt.xticks(range(len(chr_counts)), chr_counts.index, rotation=45)
plt.xlabel('Chromosome', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Top 15 Chromosomes Distribution', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# #### Strand Distribution

# In[53]:


strand_counts = df['Strand'].value_counts()
plt.bar(strand_counts.index, strand_counts.values, color=['#16a085', '#d35400'])
plt.xlabel('Strand', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Distribution of DNA Strand', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.show()


# #### Distance distribution

# In[54]:


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram
axes[0].hist(df['distance'], bins=30, color='#27ae60', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Distance', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Distribution of Distance', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Box plot by Identity
sns.boxplot(data=df, x='Identity', y='distance', palette=['#2ecc71', '#e74c3c'], ax=axes[1])
axes[1].set_title('Distance Distribution by Identity', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Identity', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Distance', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


# #### PAM sequence distribution

# In[55]:


pam_counts = df['PAM'].value_counts().head(15)

plt.figure(figsize=(14, 6))
plt.barh(pam_counts.index, pam_counts.values, color='#c0392b')
plt.xlabel('Count', fontsize=12, fontweight='bold')
plt.ylabel('PAM Sequence', fontsize=12, fontweight='bold')
plt.title('Top 15 PAM Sequences', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# #### Score vs Mismatches Scatter Plot

# In[56]:


# Scatter plot: Score vs Mismatches
plt.figure(figsize=(12, 6))
for identity in df['Identity'].unique():
    subset = df[df['Identity'] == identity]
    color = '#2ecc71' if identity == 'ON' else '#e74c3c'
    plt.scatter(subset['Mismatches'], subset['Score'], alpha=0.5, 
               label=identity, s=30, color=color)

plt.xlabel('Number of Mismatches', fontsize=12, fontweight='bold')
plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.title('Score vs Mismatches by Identity', fontsize=14, fontweight='bold')
plt.legend(title='Identity', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# #### Correlation Heatmap

# In[57]:


# Correlation heatmap for numerical features
numerical_cols = ['Score', 'Mismatches', 'distance', 'on_target_site_index']
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            fmt='.3f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()


# ## Outliers Handling
# 
# As we see in visualization there are outliers in some columns like `Score`, `distance`, and `on_target_site_index`. These outliers can significantly affect model performance and should be handled appropriately. We will use different strategies: for the `Score` column, we'll divide by 10 to normalize the scale, while for other numerical columns, we'll apply the IQR (Interquartile Range) method to cap extreme values.

# ### Check for outliers in numerical columns

# In[58]:


numerical_cols = ['Score', 'distance', 'on_target_site_index', 'Mismatches']
for col in numerical_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()


# ### Visualize outliers using box plots

# In[59]:


# Create box plots for numerical columns before handling outliers
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    sns.boxplot(data=df, y=col, ax=axes[idx], color='skyblue')
    axes[idx].set_title(f'Box Plot of {col} (Before Handling)', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(col, fontsize=11)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# In[60]:


def cap_outliers_iqr(df, column):

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers_before = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"{column}:")
    print(f"  Outliers capped: {outliers_before}")
    return df

for col in ['Score', 'distance', 'on_target_site_index']:
    df = cap_outliers_iqr(df, col)


# ### Visualize data after handling outliers

# In[61]:


# Create box plots after handling outliers
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    sns.boxplot(data=df, y=col, ax=axes[idx], color='lightgreen')
    axes[idx].set_title(f'Box Plot of {col} (After Handling)', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(col, fontsize=11)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# In[63]:


#save cleaned data
df.to_csv('DataCRISPR-Cas9-cleaned.csv', index=False)


# ## Pre-modeling

# #### Vectorization (DNA2Vec Approach)
# 
# We will apply vectorization on sequence columns: `Guide_sequence`, `Target_sequence` and `Protospacer_sequence`.
# 
# we will transform the DNA sequences into numerical vectors using a pretrained `DNA2Vec` model.
# 
# DNA2Vec is similar to Word2Vec in NLP but instead of words, it learns vector representations for k-mers.
# Each k-mer gets a fixed-length embedding that captures its biological and contextual meaning based on large genomic corpora.
# > **k-mer** here meaning break sequence into substrings of length k (if k=3 → "ACG", "CGT", …) and count frequencies.
# 
# The flow is :
# * Splitting sequences into overlapping k-mers.
# * Apply DNA2Vec model for each k-mer.
# * Averaging all k-mer vectors in the sequence to get one dense vector that summarizes the whole sequence.
# 

# In[64]:


from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[66]:


model = KeyedVectors.load_word2vec_format("dna2vec.w2v", binary=False)

def sequence_to_vector(seq, k=3):
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    vectors = [model[kmer] for kmer in kmers if kmer in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


# Each value in sequences columns is a NumPy array after vectorization, so we need to split it into individual columns (one per embedding dimension). To do that we will use `.tolist()`.

# In[67]:


df['Target_vec'] = df['Target_sequence'].apply(sequence_to_vector)
df['Proto_vec']  = df['Protospacer_sequence'].apply(sequence_to_vector)
df['Guide_vec'] = df['Guide_sequence'].apply(sequence_to_vector)


target_vec_df = pd.DataFrame(df['Target_vec'].tolist(), columns=[f"Target_emb_{i}" for i in range(model.vector_size)])
proto_vec_df  = pd.DataFrame(df['Proto_vec'].tolist(),  columns=[f"Proto_emb_{i}"  for i in range(model.vector_size)])
guide_vec_df  = pd.DataFrame(df['Guide_vec'].tolist(),  columns=[f"Guide_emb_{i}"  for i in range(model.vector_size)])


# No need for original columns so let's drop them.
# 

# In[68]:


df = df.drop(columns=["Guide_sequence", "Target_sequence", "Protospacer_sequence", 
                              "Guide_vec", "Target_vec", "Proto_vec"], errors="ignore")

# Combine embeddings + other features
embed_df = pd.concat([df.reset_index(drop=True),
                    guide_vec_df,
                    target_vec_df,
                    proto_vec_df], axis=1)


# #### Data encoding

# In[69]:


categorical_cols = embed_df.select_dtypes(include=['object']).columns
categorical_cols


# In[70]:


for col in categorical_cols:
    le = LabelEncoder()
    embed_df[col] = le.fit_transform(embed_df[col])    


# In[71]:


embed_df


# #### Standardize the features

# In[72]:


X = embed_df.drop(columns=['Identity'])
y = embed_df['Identity']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# #### PCA
# 
# PCA used for dimensionality reduction that transforms the original features into a set of linearly uncorrelated components. This helps to:
# - Reduce computational complexity
# - Remove multicollinearity

# In[73]:


pca_full = PCA()
pca_full.fit(X_scaled)

# Calculate cumulative explained variance
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"Total number of components: {len(explained_variance)}")
print(f"\nExplained variance by first 5 components:")
for i in range(min(5, len(explained_variance))):
    print(f"  PC{i+1}: {explained_variance[i]:.4f} ({explained_variance[i]*100:.2f}%)")

print(f"\nCumulative variance explained:")
for threshold in [0.80, 0.90, 0.95, 0.99]:
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    print(f"  {threshold*100:.0f}% variance: {n_components} components")


# In[75]:


pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca_3d, columns=['PC1', 'PC2', 'PC3'])
pca_df['Identity'] = y.values

print(f"Variance explained by 3 components: {pca_3d.explained_variance_ratio_.sum():.4f} ({pca_3d.explained_variance_ratio_.sum()*100:.2f}%)")
print(f"  PC1: {pca_3d.explained_variance_ratio_[0]:.4f} ({pca_3d.explained_variance_ratio_[0]*100:.2f}%)")
print(f"  PC2: {pca_3d.explained_variance_ratio_[1]:.4f} ({pca_3d.explained_variance_ratio_[1]*100:.2f}%)")
print(f"  PC3: {pca_3d.explained_variance_ratio_[2]:.4f} ({pca_3d.explained_variance_ratio_[2]*100:.2f}%)")


# #### Data Spliting
# 
# Given the large size of the dataset, a single train/test split is sufficient to obtain reliable evaluation results. Cross-validation could still be used for extra rigor, but in practice it is often unnecessary for such large datasets.

# In[76]:


# Use PCA-transformed data for modeling
x = pca_df.drop(columns=['Identity'])
y = pca_df['Identity'] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)


# In[77]:


print("Training set size:", x_train.shape)
print("Testing set size:", x_test.shape)


# In[78]:


y.value_counts(normalize=True)


# In[ ]:


x_train


# ## Modeling

# In[79]:


classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))
print("Class Weights:", class_weights)


# After calculating the class weights, we can see that our dataset is extremely imbalanced.
# The class weight ratio is roughly 1 : 48, which means that almost 98% of the samples belong to class 1, while only about 2% belong to class 0.

# This makes the classification task highly challenging, as the model can easily achieve high accuracy by always predicting the majority class, but that would provide no real biological insight.
# 
# Normally, one might try to fix this issue using oversampling (e.g., SMOTE) or undersampling techniques to balance the data.
# 
# The data represents biological sequences (e.g., guide RNA–target pairs), which have biological meaning and structure.

# In[83]:


from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score


# Why we chose XGBClassifier for this project?
# because:
# 
# * Handles Imbalanced Data Well: With parameters like `scale_pos_weight` or using class weights, XGBoost can effectively learn from minority classes without being biased towards the majority class.
# 
# * XGBoost includes regularization (L1 & L2) which helps prevent overfitting, especially important when the dataset is large but the minority class is small.

# In[84]:


model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=class_weights[0] / class_weights[1],  # handle imbalance
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)


# In[85]:


model.fit(x_train, y_train)


# In[87]:


y_test_pred = model.predict(x_test)
y_test_prob = model.predict_proba(x_test)[:, 1]


# In[88]:


print(classification_report(y_test, y_test_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_test_prob))


# The model achieved high overall accuracy (0.98), but it completely failed to detect the minority class (class 1), as shown by the zero precision, recall, and F1-score for that class. This indicates a strong class imbalance problem, where the model is biased toward predicting the majority class (class 0).
# 
# Although the accuracy looks excellent, it is misleading because the model does not correctly identify any positive samples. To improve performance on class 1, techniques such as SMOTE (oversampling), class weighting, or threshold adjustment should be applied.

# In[93]:


from imblearn.over_sampling import SMOTE


# In[94]:


sm = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_res, y_train_res = sm.fit_resample(x_train, y_train)


# In[95]:


model.fit(X_train_res, y_train_res)


# In[ ]:


y_test_pred = model.predict(x_test)
y_test_prob = model.predict_proba(x_test)[:, 1]


# In[96]:


print(classification_report(y_test, y_test_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_test_prob))


# The model achieved perfect performance on the test set, with precision, recall, and F1-score all equal to 1.00 for both classes, and a ROC-AUC of 1.0. While this indicates excellent predictive ability, the test set is quite small (only 157 samples, with 3 positives), which might lead to overly optimistic results.
# 
# To ensure the model’s robustness and generalization, we will apply K-Fold cross-validation on the training data. This approach will help us evaluate the model’s consistency and confirm that its performance is not dependent on a specific data split.

# In[97]:


from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# In[98]:


pipeline = Pipeline([
    ('smote', SMOTE(random_state=42, sampling_strategy=0.5)),
    ('xgb', XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    ))
])


# In[99]:


param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__subsample': [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0]
}


# In[100]:


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=2
)


# In[101]:


grid.fit(x_train, y_train)


# In[102]:


print("Best Parameters:", grid.best_params_)
print("Best ROC-AUC Score:", grid.best_score_)


# In[103]:


y_pred = grid.best_estimator_.predict(x_test)
y_proba = grid.best_estimator_.predict_proba(x_test)[:, 1]


# In[104]:


print(classification_report(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_proba))


# Model Evaluation Report
# 
# The XGBoost classifier achieved excellent overall performance, with 99% accuracy and a perfect ROC-AUC score of 1.0, indicating that the model can clearly distinguish between the two classes.
# 
# For class 1 (the minority class), the recall is 1.00, meaning the model successfully detected all positive cases without missing any. However, the precision is 0.75, which indicates that some of the samples predicted as class 1 were actually class 0 (false positives).
# 
# This usually happens when the dataset is imbalanced — the number of samples in class 1 is much smaller compared to class 0. In this case, there were only 3 positive samples out of 157. The model tends to overpredict the minority class slightly to ensure high recall, which reduces precision a bit.
# 
# > In many real-world cases (such as medical diagnosis or rare event detection), recall is more important than precision, as missing a positive case can be more costly than a few false alarms.

# In[106]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    model,
    x_test,
    y_test,
    cmap="Blues",
    values_format="d"
)

plt.title("Confusion Matrix")
plt.show()


# In[ ]:


import joblib

best_model = grid.best_estimator_
joblib.dump(best_model, "best_xgb_model.pkl")


# In[108]:


y_test_df = pd.DataFrame({'True_Label': y_test, 'Predicted_Label': y_pred, 'Predicted_Probability': y_proba})
y_test_df.to_csv('testset_predictions.csv', index=False)


# In[ ]:




