import pytest
import os

# Função para verificar se o arquivo CSV foi criado corretamente
def test_csv_file_created():
    assert os.path.isfile('evaluation_results.csv') == True

# Função para verificar se o arquivo PNG da curva ROC foi criado corretamente
def test_png_file_created():
    assert os.path.isfile('curva_roc.png') == True

# Executar os testes usando Pytest
if __name__ == "__main__":
    pytest.main()
