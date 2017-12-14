import pandas as pd

df1 = pd.DataFrame([[23, 'Moscow', 'Msk', ''],
                    [34, 'Obninsk', 'Msk', 'Msk'],
                    [56, '', '', 'Spb'],
                    [17, 'Tula', 'Spb', '']],
                   columns=['ID', 'City', 'Region', '2City'])
df2 = pd.DataFrame([['Msk', 'Msk'],
                    ['Spb', 'Spb'],
                    ['Tula', 'Msk'],
                    ['Moscow', 'Msk']],
                   columns=['City', 'Office'])

# df = pd.concat([df1.loc[df1[x].isin(df2['City']), x] for x in ['City', 'Region', '2City']])
df = pd.concat([df1.loc[df1[x].isin(df2['City']), x] for x in ['City']])

df1['Join'] = df.groupby(df.index).first()

output = df1.merge(df2, left_on='Join', right_on='City', how='right')
df1['Office'] = output['Office']
df1.drop('Join', axis=1, inplace=True)

print(output)
print(df1)
