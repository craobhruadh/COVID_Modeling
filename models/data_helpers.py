import pandas as pd

POPULATION_OF_LA = 10200000


def read_file(filename='./data/LA_County_Covid19_cases_deaths_date_table.csv'):
    df = pd.read_csv(
        filename,
        parse_dates=['date_use'],
        index_col=0
    )
    df.sort_values(by='date_use', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Assume 'infected' is the same as the avg_cases
    df['suceptible'] = POPULATION_OF_LA - df['total_cases'] - df['total_deaths']
    df['recovered'] = POPULATION_OF_LA - df['suceptible'] - df['avg_cases']
    df[['S', 'I', 'R']] = df[['suceptible', 'avg_cases', 'recovered']].apply(lambda x: x/POPULATION_OF_LA, axis=1)

    df['I'].fillna(value=0, axis=0, inplace=True)
    df['R'].fillna(value=0, axis=0, inplace=True)
    return df


def prep_data(susceptible,
              infected,
              recovered,
              n_people=POPULATION_OF_LA):
    s = susceptible / n_people
    i = infected / n_people
    r = recovered / n_people
    return s, i, r
