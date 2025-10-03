-- database/schema_sqlserver.sql
-- SQL Server Schema for SmartPortfolio

USE master;
GO

-- Create database if not exists
IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'SmartPortfolio')
BEGIN
    CREATE DATABASE SmartPortfolio;
END
GO

USE SmartPortfolio;
GO

-- Create schemas
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'market')
    EXEC('CREATE SCHEMA market');
GO

IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'fundamental')
    EXEC('CREATE SCHEMA fundamental');
GO

IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'institutional')
    EXEC('CREATE SCHEMA institutional');
GO

IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'macro')
    EXEC('CREATE SCHEMA macro');
GO

IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'portfolio')
    EXEC('CREATE SCHEMA portfolio');
GO

IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'ml_models')
    EXEC('CREATE SCHEMA ml_models');
GO

-- ============================================
-- MARKET DATA TABLES
-- ============================================

-- S&P 500 Tickers
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'market.sp500_tickers') AND type = 'U')
BEGIN
    CREATE TABLE market.sp500_tickers (
        ticker VARCHAR(10) PRIMARY KEY,
        company_name VARCHAR(255),
        sector VARCHAR(100),
        industry VARCHAR(150),
        added_date DATE,
        removed_date DATE,
        is_active BIT DEFAULT 1,
        market_cap BIGINT,
        created_at DATETIME DEFAULT GETDATE(),
        updated_at DATETIME DEFAULT GETDATE()
    );
END
GO

-- Daily Price Data
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'market.daily_prices') AND type = 'U')
BEGIN
    CREATE TABLE market.daily_prices (
        id INT IDENTITY(1,1) PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        [open] DECIMAL(12, 4),
        high DECIMAL(12, 4),
        low DECIMAL(12, 4),
        [close] DECIMAL(12, 4),
        adj_close DECIMAL(12, 4),
        volume BIGINT,
        created_at DATETIME DEFAULT GETDATE(),
        CONSTRAINT unique_ticker_date UNIQUE (ticker, date),
        CONSTRAINT fk_ticker FOREIGN KEY (ticker) REFERENCES market.sp500_tickers(ticker)
    );
    
    CREATE INDEX idx_prices_ticker ON market.daily_prices(ticker);
    CREATE INDEX idx_prices_date ON market.daily_prices(date);
    CREATE INDEX idx_prices_ticker_date ON market.daily_prices(ticker, date);
END
GO

-- Dividends
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'market.dividends') AND type = 'U')
BEGIN
    CREATE TABLE market.dividends (
        id INT IDENTITY(1,1) PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        ex_date DATE NOT NULL,
        payment_date DATE,
        amount DECIMAL(10, 4),
        frequency VARCHAR(20),
        created_at DATETIME DEFAULT GETDATE(),
        CONSTRAINT unique_ticker_dividend UNIQUE (ticker, ex_date),
        CONSTRAINT fk_ticker_div FOREIGN KEY (ticker) REFERENCES market.sp500_tickers(ticker)
    );
END
GO

-- Stock Splits
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'market.splits') AND type = 'U')
BEGIN
    CREATE TABLE market.splits (
        id INT IDENTITY(1,1) PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        split_ratio VARCHAR(20),
        created_at DATETIME DEFAULT GETDATE(),
        CONSTRAINT unique_ticker_split UNIQUE (ticker, date),
        CONSTRAINT fk_ticker_split FOREIGN KEY (ticker) REFERENCES market.sp500_tickers(ticker)
    );
END
GO

-- ============================================
-- FUNDAMENTAL DATA TABLES
-- ============================================

-- Financial Ratios
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'fundamental.financial_ratios') AND type = 'U')
BEGIN
    CREATE TABLE fundamental.financial_ratios (
        id INT IDENTITY(1,1) PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        date DATE NOT NULL,
        pe_ratio DECIMAL(10, 2),
        pb_ratio DECIMAL(10, 2),
        ps_ratio DECIMAL(10, 2),
        peg_ratio DECIMAL(10, 2),
        roe DECIMAL(10, 4),
        roa DECIMAL(10, 4),
        debt_equity DECIMAL(10, 4),
        current_ratio DECIMAL(10, 4),
        quick_ratio DECIMAL(10, 4),
        gross_margin DECIMAL(10, 4),
        operating_margin DECIMAL(10, 4),
        net_margin DECIMAL(10, 4),
        created_at DATETIME DEFAULT GETDATE(),
        CONSTRAINT unique_ticker_ratio_date UNIQUE (ticker, date),
        CONSTRAINT fk_ticker_ratio FOREIGN KEY (ticker) REFERENCES market.sp500_tickers(ticker)
    );
    
    CREATE INDEX idx_ratios_ticker ON fundamental.financial_ratios(ticker);
    CREATE INDEX idx_ratios_date ON fundamental.financial_ratios(date);
END
GO

-- ============================================
-- PORTFOLIO TRACKING TABLES
-- ============================================

-- Portfolio Positions
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'portfolio.positions') AND type = 'U')
BEGIN
    CREATE TABLE portfolio.positions (
        id INT IDENTITY(1,1) PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        entry_date DATE NOT NULL,
        entry_price DECIMAL(12, 4),
        shares INT,
        investment DECIMAL(15, 2),
        current_price DECIMAL(12, 4),
        current_value DECIMAL(15, 2),
        unrealized_pnl DECIMAL(15, 2),
        target_price DECIMAL(12, 4),
        stop_loss DECIMAL(12, 4),
        time_horizon VARCHAR(20),
        is_active BIT DEFAULT 1,
        exit_date DATE,
        exit_price DECIMAL(12, 4),
        realized_pnl DECIMAL(15, 2),
        created_at DATETIME DEFAULT GETDATE(),
        updated_at DATETIME DEFAULT GETDATE(),
        CONSTRAINT fk_ticker_position FOREIGN KEY (ticker) REFERENCES market.sp500_tickers(ticker)
    );
END
GO

-- Transactions
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'portfolio.transactions') AND type = 'U')
BEGIN
    CREATE TABLE portfolio.transactions (
        id INT IDENTITY(1,1) PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        transaction_date DATE NOT NULL,
        transaction_type VARCHAR(10),
        shares INT,
        price DECIMAL(12, 4),
        commission DECIMAL(10, 2),
        total_amount DECIMAL(15, 2),
        notes VARCHAR(MAX),
        created_at DATETIME DEFAULT GETDATE(),
        CONSTRAINT fk_ticker_transaction FOREIGN KEY (ticker) REFERENCES market.sp500_tickers(ticker)
    );
END
GO

-- ============================================
-- ML MODELS & PREDICTIONS TABLES
-- ============================================

-- Return Predictions
IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'ml_models.return_predictions') AND type = 'U')
BEGIN
    CREATE TABLE ml_models.return_predictions (
        id INT IDENTITY(1,1) PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        prediction_date DATE NOT NULL,
        horizon VARCHAR(20),
        predicted_return DECIMAL(10, 4),
        confidence_lower DECIMAL(10, 4),
        confidence_upper DECIMAL(10, 4),
        model_name VARCHAR(50),
        model_version VARCHAR(20),
        actual_return DECIMAL(10, 4),
        prediction_error DECIMAL(10, 4),
        created_at DATETIME DEFAULT GETDATE(),
        CONSTRAINT unique_ticker_pred UNIQUE (ticker, prediction_date, horizon),
        CONSTRAINT fk_ticker_pred FOREIGN KEY (ticker) REFERENCES market.sp500_tickers(ticker)
    );
    
    CREATE INDEX idx_predictions_ticker ON ml_models.return_predictions(ticker);
    CREATE INDEX idx_predictions_date ON ml_models.return_predictions(prediction_date);
END
GO

PRINT '✅ SQL Server schema created successfully!';
GO