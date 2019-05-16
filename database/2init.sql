\c share_ai
CREATE TABLE account(
username    VARCHAR(70) PRIMARY KEY,
password    VARCHAR(100) not null,
creation_time_stamp    TIMESTAMP not null,
role    VARCHAR(10) not null,
CONSTRAINT check_role
   CHECK (role = 'user' OR role = 'root')
);
CREATE TABLE user_actions(
action_name   VARCHAR(6) PRIMARY KEY,
description   VARCHAR(50) not null
);


CREATE TABLE account_log(
log_id    SERIAL PRIMARY KEY,
username    VARCHAR(70),
description   VARCHAR(50) not null,
log_timestamp    TIMESTAMP not null,
CONSTRAINT log_account_username_fkey FOREIGN KEY (username)
      REFERENCES account (username) MATCH SIMPLE
      ON UPDATE NO ACTION ON DELETE NO ACTION
);

GRANT USAGE ON SEQUENCE account_log_log_id_seq to share_ai_api;

CREATE TABLE user_account(
username    VARCHAR(70) PRIMARY KEY,
first_name  VARCHAR(70) not null,
last_name   VARCHAR(70) not null,
country     VARCHAR(2) not null,
state_or_province   VARCHAR(2) not null,
city        VARCHAR(30) not null,
street_address VARCHAR(30) not null,
profile_picture VARCHAR(60),
email   VARCHAR(30) not null,
CONSTRAINT user_account_username_fkey FOREIGN KEY (username)
    REFERENCES account (username) MATCH SIMPLE
    ON UPDATE NO ACTION ON DELETE NO ACTION
);
CREATE TABLE organization(
name    VARCHAR(70) PRIMARY KEY,
description VARCHAR (50) not null,
affiliation VARCHAR(50) not null
);
-- noinspection SqlNoDataSourceInspection

CREATE TABLE organization_affiliate(
username    VARCHAR(70),
organization_name    VARCHAR(70),
affiliate_type    VARCHAR(6) not null,
PRIMARY KEY(username, organization_name),
CONSTRAINT organization_affiliate_username_fkey FOREIGN KEY (username)
    REFERENCES account (username) MATCH SIMPLE
    ON UPDATE NO ACTION ON DELETE NO ACTION,
CONSTRAINT organization_affiliate_name_fkey FOREIGN KEY (organization_name)
    REFERENCES organization(name) MATCH SIMPLE
    ON UPDATE NO ACTION ON DELETE NO ACTION
);
CREATE TABLE model(
name    VARCHAR(70) PRIMARY KEY,
organization_name   VARCHAR(70),
description     VARCHAR(50) not null,
implementation_language VARCHAR(70) not null,
implementation_framework VARCHAR(70) not null,
model_domain    VARCHAR(70) not null,
model_subdomain VARCHAR(70) not null,
task    VARCHAR(50) not null,
CONSTRAINT model_organization_name_fkey FOREIGN KEY (organization_name)
    REFERENCES organization(name) MATCH SIMPLE
    ON UPDATE NO ACTION ON DELETE NO ACTION
);
CREATE TABLE version(
name    VARCHAR(70) not null REFERENCES model(name),
version_id VARCHAR(10) not null,
description VARCHAR(50) not null,
storage_url VARCHAR(50) not null,
price   FLOAT(2) not null,
creation_timestamp TIMESTAMP not null,
implementation_framework_version VARCHAR(10) not null,
PRIMARY KEY(name, version_id)
);
CREATE TABLE transactions(
transaction_id  SERIAL PRIMARY KEY,
username    VARCHAR(70) not null,
version_id  VARCHAR(10) not null,
model_name    VARCHAR(70) not null,
transaction_timestamp TIMESTAMP not null,
CONSTRAINT transactions_username_fkey FOREIGN KEY (username)
    REFERENCES user_account(username) MATCH SIMPLE
    ON UPDATE NO ACTION ON DELETE NO ACTION,
CONSTRAINT version_id_name_fkey FOREIGN KEY (version_id, model_name)
    REFERENCES version(version_id, name) MATCH SIMPLE
    ON UPDATE NO ACTION ON DELETE NO ACTION
);

GRANT USAGE ON SEQUENCE transactions_transaction_id_seq to share_ai_api;

CREATE TABLE invitation(
organization    VARCHAR(70) not null,
recipient_username  VARCHAR(70) not null,
inviter_username    VARCHAR(70) not null,
status  VARCHAR(10) not null,
PRIMARY KEY (organization, recipient_username),
CONSTRAINT invitation_organization_fkey FOREIGN KEY (organization)
    REFERENCES organization(name) MATCH SIMPLE
    ON UPDATE NO ACTION ON DELETE NO ACTION,
CONSTRAINT recipient_username_fkey FOREIGN KEY (recipient_username)
    REFERENCES account(username) MATCH SIMPLE
    ON UPDATE NO ACTION ON DELETE NO ACTION,
CONSTRAINT inviter_username_fkey FOREIGN KEY (inviter_username)
    REFERENCES account(username) MATCH SIMPLE
    ON UPDATE NO ACTION ON DELETE NO ACTION
);

-----TRIGGERS

CREATE FUNCTION log_account_action() RETURNS trigger AS $account_stamp$
    BEGIN

        INSERT INTO account_log (username, description, log_timestamp) VALUES (NEW.username , TG_ARGV[0], now());
        RETURN NEW;
    END;
$account_stamp$ LANGUAGE plpgsql;

CREATE TRIGGER log_update_account
    AFTER INSERT ON account
    FOR EACH ROW
    EXECUTE PROCEDURE log_account_action('Created Account');

CREATE TRIGGER account_update
    AFTER UPDATE on user_account
    FOR EACH ROW
    EXECUTE PROCEDURE log_account_action('Edited account info');

CREATE TRIGGER transaction_made
    AFTER INSERT on transactions
    FOR EACH ROW
    EXECUTE PROCEDURE log_account_action('Purchased model');



-------------


GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO share_ai_api;

INSERT INTO account values ('root', '$2a$10$n3W8ekUIOgR0pClXgeEKMOX3eidoDISQPoTN69V0DaYCIdL5XBeBe', current_timestamp, 'root');